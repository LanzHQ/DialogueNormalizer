-- @description Dialogue Normalizer - phrase-by-phrase dialogue leveling (EBU R128)
-- @version 1.0
-- @author Ilya Lavrin
--
-- ALV Dialogue Normalizer   -   -23 LUFS, phrase by phrase, no settings
--
-- NORMALIZE: finds every spoken phrase, measures it on its own (K-weighted,
-- EBU R128) and brings it to -23 LUFS as a whole, so dynamics INSIDE a phrase
-- are never touched - no riding, no flattening, no pumping. Gain changes happen
-- only in pauses, at their quietest point, where they are inaudible.
-- Breaths, room tone and noise are recognised (they lack syllabic modulation)
-- and are never normalized on their own - they keep the gain of the phrase
-- next to them.
--
-- SPLIT AT SILENCE: cuts the recording into one clip per phrase and removes the
-- room tone in between, with generous pads, fades and decay-tail tracking so
-- word endings and breaths are not clipped off. Recommended BEFORE normalizing:
-- the leveling result is identical either way, but with the silence gone there
-- is no background noise left to be lifted with the quiet lines.
--
-- Scope: selected items; if none, all items on selected tracks.
-- Re-running is idempotent: analysis always reads the raw source.

local r = reaper

local EXT        = 'ALV_NORM'
local REF        = -23.0  -- reference used for measuring (not the goal)
local MAX_BOOST  = 30.0   -- dB
local MAX_CUT    = 20.0   -- dB
local CEILING    = -1.0   -- dBFS, sample peak allowed after gain
local SHORT_PH   = 0.60   -- s, below this a phrase follows its neighbours
local SHORT_DEV  = 3.0    -- dB, how far a short phrase may differ from them

local BLK        = 0.01   -- s, detection resolution (peak envelope)
local MERGE_GAP  = 0.35   -- s, pauses shorter than this stay inside one phrase
local MIN_PHRASE = 0.25   -- s
local RAMP_MAX   = 0.40   -- s, longest gain transition inside a pause
local CHUNK_PTS  = 30000  -- peak points per API call (~5 min)

local PAD_LEAD   = 0.15   -- s, kept before each phrase when splitting
local PAD_TRAIL  = 0.35   -- s, kept after each phrase when splitting
local TAIL_MAX   = 0.60   -- s, how far a decaying tail is followed
local TAIL_OVER  = 4.0    -- dB above the noise floor that still counts as tail

local function db2lin(db) return 10 ^ (db / 20) end
local function lin2db(v) return 20 * math.log(v, 10) end
local function clamp(v, lo, hi) return v < lo and lo or (v > hi and hi or v) end

local function trace(fmt, ...)
  if _G.ALV_TRACE then _G.ALV_TRACE(string.format(fmt, ...)) end
end

local function getnum(key, default)
  local v = tonumber(r.GetExtState(EXT, key))
  return v or default
end

local hide_curves = r.GetExtState(EXT, 'hide_curves') == '1'
local target      = getnum('target', -23.0)    -- LUFS goal, user editable
local evenness    = getnum('evenness', 85.0)   -- % of the level spread removed

-- loudness (LUFS) of a source region, via REAPER's own R128 analysis.
-- A gain of exactly 1.0 means REAPER could not measure it (the region is below
-- R128's absolute gate) - that must not be read as "already on target".
local function region_loudness(src, t0, t1)
  local g = r.CalculateNormalization(src, 0, REF, t0, t1)
  if not g or g <= 0 or math.abs(g - 1.0) < 1e-9 then return nil end
  return REF - lin2db(g)
end

-- duration-weighted loudness of a set of phrases, with R128's relative gate
local function integrated(list, with_gain)
  local function mean(thresh)
    local acc, dur = 0, 0
    for _, p in ipairs(list) do
      local l = p.lufs + (with_gain and p.gain or 0)
      if not thresh or l > thresh then
        acc = acc + p.dur * 10 ^ (l / 10); dur = dur + p.dur
      end
    end
    return dur > 0 and 10 * math.log(acc / dur, 10) or nil
  end
  local m = mean(nil)
  if not m then return nil end
  return mean(m - 10) or m
end

local function collect_items()
  local items = {}
  local n = r.CountSelectedMediaItems(0)
  if n > 0 then
    for i = 0, n - 1 do items[#items + 1] = r.GetSelectedMediaItem(0, i) end
    return items
  end
  for t = 0, r.CountSelectedTracks(0) - 1 do
    local tr = r.GetSelectedTrack(0, t)
    for i = 0, r.CountTrackMediaItems(tr) - 1 do
      items[#items + 1] = r.GetTrackMediaItem(tr, i)
    end
  end
  return items
end

---------------------------------------------------------------- analysis

-- peak envelope of the item, one value per BLK seconds, in dBFS.
-- Uses REAPER's native peak reader (full-rate accurate, ~1000x faster than
-- reading samples). NOTE: its start time is PROJECT time, not item time.
-- Returns nil if peaks are unavailable.
local function peak_envelope(take, pos, len, nch, exec)
  local npts = math.floor(len / BLK)
  if npts < 8 then return nil end
  local ch = math.min(math.max(nch, 1), 2)
  local env, nonzero, done = {}, 0, 0
  while done < npts do
    local want = math.min(CHUNK_PTS, npts - done)
    local buf = r.new_array(want * ch * 2)
    buf.clear()
    local ret = exec(r.GetMediaItemTake_Peaks, take, 1 / BLK, pos + done * BLK,
                     ch, want, 0, buf)
    if not ret then return nil end
    local got = ret & 0xfffff
    if got < 1 then return nil end
    for i = 1, got do
      local mx = 0
      for c = 0, ch - 1 do
        local a = math.abs(buf[(i - 1) * ch + c + 1])
        local b = math.abs(buf[got * ch + (i - 1) * ch + c + 1])
        if a > mx then mx = a end
        if b > mx then mx = b end
      end
      if mx > 1e-7 then nonzero = nonzero + 1; env[done + i] = lin2db(mx)
      else env[done + i] = -120 end
    end
    done = done + got
    if got < want then break end
  end
  if #env < 8 or nonzero < 4 then return nil end
  return env
end

-- split the item into phrases; returns list of {s,e,...} in item time + stats
local function find_phrases(env)
  local n = #env
  local live = {}
  for i = 1, n do if env[i] > -85 then live[#live + 1] = env[i] end end
  if #live < 20 then return nil end        -- under 0.2 s of audible content
  table.sort(live)
  local function pct(q) return live[clamp(math.floor(#live * q), 1, #live)] end
  local speech_ref = pct(0.85)             -- typical speech peak level
  local quiet_ref  = pct(0.10)             -- room tone / noise floor
  if speech_ref < -60 then return nil end  -- nothing but near-silence here
  local gate = math.max(speech_ref - 24, math.min(quiet_ref + 6, speech_ref - 12))
  local hyst = 8

  local raw, open, st = {}, false, 0
  for i = 1, n do
    if not open and env[i] >= gate then
      open, st = true, i
    elseif open and env[i] < gate - hyst then
      open = false
      raw[#raw + 1] = { s = (st - 1) * BLK, e = i * BLK }
    end
  end
  if open then raw[#raw + 1] = { s = (st - 1) * BLK, e = n * BLK } end

  local merged = {}
  for _, sg in ipairs(raw) do
    local p = merged[#merged]
    if p and (sg.s - p.e) < MERGE_GAP then p.e = sg.e
    else merged[#merged + 1] = { s = sg.s, e = sg.e } end
  end

  -- level, modulation and classification: speech vs breath / noise / click.
  -- Speech is strongly modulated by syllables (wide spread between its loud and
  -- quiet blocks); breaths, hum and room tone are smooth and low.
  local out = {}
  for _, sg in ipairs(merged) do
    local i0 = math.max(1, math.floor(sg.s / BLK) + 1)
    local i1 = math.min(n, math.ceil(sg.e / BLK))
    local acc, cnt, pk, vals = 0, 0, -120, {}
    for i = i0, i1 do
      if env[i] > -110 then
        acc = acc + 10 ^ (env[i] / 10); cnt = cnt + 1
        vals[#vals + 1] = env[i]
      end
      if env[i] > pk then pk = env[i] end
    end
    local lvl = cnt > 0 and 10 * math.log(acc / cnt, 10) or -120
    table.sort(vals)
    local mod = #vals >= 4
      and (vals[math.ceil(#vals * 0.9)] - vals[math.ceil(#vals * 0.25)]) or 0
    local dur = sg.e - sg.s
    local speech
    if pk > speech_ref - 12 then
      speech = dur >= MIN_PHRASE            -- clearly loud: speech
    elseif dur >= 0.6 and pk > speech_ref - 30 then
      speech = mod >= 6                     -- quiet: only if syllable-modulated
    else
      speech = false
    end
    out[#out + 1] = { s = sg.s, e = sg.e, lvl = lvl, pk = pk, mod = mod,
                      dur = dur, speech = speech }
  end
  trace('  speech_ref=%.1f quiet_ref=%.1f gate=%.1f -> %d raw, %d merged',
    speech_ref, quiet_ref, gate, #raw, #merged)
  return out, { gate = gate, speech_ref = speech_ref, quiet_ref = quiet_ref }
end

-- the quietest moment inside a pause: where a gain change is inaudible
local function quiet_point(env, t0, t1, gate)
  local i0 = math.max(1, math.floor(t0 / BLK) + 1)
  local i1 = math.min(#env, math.ceil(t1 / BLK))
  if i1 <= i0 then return (t0 + t1) * 0.5 end
  local thr = gate - 12
  local bs, be, cs, best = nil, nil, nil, 0
  for i = i0, i1 do
    if env[i] < thr then
      cs = cs or i
      if i - cs + 1 > best then best, bs, be = i - cs + 1, cs, i end
    else cs = nil end
  end
  if bs then return ((bs + be) * 0.5 - 0.5) * BLK end
  local mi, mv = i0, math.huge
  for i = i0, i1 do if env[i] < mv then mv, mi = env[i], i end end
  return (mi - 0.5) * BLK
end

-- keep-regions for splitting: speech + decay tail + pads, overlaps merged
local function speech_regions(phrases, stats, env, len)
  local floor_db = stats.quiet_ref + TAIL_OVER
  local out = {}
  for idx, p in ipairs(phrases) do
    if p.speech then
      -- follow the decaying tail while it stays above the noise floor
      local e = p.e
      local nxt = len
      for j = idx + 1, #phrases do
        if phrases[j].speech then nxt = phrases[j].s break end
      end
      local limit = math.min(p.e + TAIL_MAX, nxt - 0.05, len)
      local i = math.floor(p.e / BLK) + 1
      while i <= #env do
        local bt = i * BLK
        if bt > limit or env[i] <= floor_db then break end
        e = bt
        i = i + 1
      end
      local s = clamp(p.s - PAD_LEAD, 0, len)
      local ee = clamp(e + PAD_TRAIL, 0, len)
      local prev = out[#out]
      if prev and s <= prev.e then
        prev.e = math.max(prev.e, ee)
      else
        out[#out + 1] = { s = s, e = ee, onset = p.s }
      end
    end
  end
  return out
end

---------------------------------------------------------------- envelopes

local function set_env_visible(take, visible)
  local env = r.GetTakeEnvelopeByName(take, 'Volume')
  if not env then return end
  local ok, chunk = r.GetEnvelopeStateChunk(env, '', false)
  if not ok then return end
  local newchunk = chunk:gsub('\nVIS %d', '\nVIS ' .. (visible and '1' or '0'), 1)
  if newchunk ~= chunk then r.SetEnvelopeStateChunk(env, newchunk, false) end
end

local function ensure_take_vol_env(item, take)
  local env = r.GetTakeEnvelopeByName(take, 'Volume')
  if env then return env end
  r.SelectAllMediaItems(0, false)
  r.SetMediaItemSelected(item, true)
  r.Main_OnCommand(40693, 0) -- Take: Toggle take volume envelope
  return r.GetTakeEnvelopeByName(take, 'Volume')
end

local function clear_take_env(take)
  local env = r.GetTakeEnvelopeByName(take, 'Volume')
  if not env then return end
  local mode = r.GetEnvelopeScalingMode(env)
  local unity = r.ScaleToEnvelopeMode(mode, 1.0)
  r.DeleteEnvelopePointRange(env, -1e6, 1e6)
  if r.CountEnvelopePoints(env) > 0 then
    r.SetEnvelopePoint(env, 0, 0.0, unity, 0, 0, false, true)
  else
    r.InsertEnvelopePoint(env, 0, unity, 0, 0, false, true)
  end
  r.Envelope_SortPoints(env)
end

---------------------------------------------------------------- item helpers

-- read the raw source with any existing gain/envelope temporarily neutralized
local function analyse_raw(item, take, len, nch, exec)
  local itemvol = r.GetMediaItemInfo_Value(item, 'D_VOL')
  local takevol = r.GetMediaItemTakeInfo_Value(take, 'D_VOL')
  r.SetMediaItemInfo_Value(item, 'D_VOL', 1)
  r.SetMediaItemTakeInfo_Value(take, 'D_VOL', takevol >= 0 and 1 or -1)
  local tenv, envchunk = r.GetTakeEnvelopeByName(take, 'Volume'), nil
  if tenv then
    local okc, chunk = r.GetEnvelopeStateChunk(tenv, '', false)
    if okc and chunk:find('\nACT 1') then
      envchunk = chunk
      r.SetEnvelopeStateChunk(tenv, chunk:gsub('\nACT 1', '\nACT 0', 1), false)
    end
  end
  local env = peak_envelope(take, r.GetMediaItemInfo_Value(item, 'D_POSITION'),
                            len, nch, exec)
  r.SetMediaItemInfo_Value(item, 'D_VOL', itemvol)
  r.SetMediaItemTakeInfo_Value(take, 'D_VOL', takevol)
  if envchunk then r.SetEnvelopeStateChunk(tenv, envchunk, false) end
  return env, takevol
end

local function item_basics(item, rep)
  local take = r.GetActiveTake(item)
  if not take or r.TakeIsMIDI(take) then rep.skipped = rep.skipped + 1; return end
  if r.GetMediaItemInfo_Value(item, 'B_MUTE') == 1 then
    rep.skipped = rep.skipped + 1; return
  end
  local src = r.GetMediaItemTake_Source(take)
  local rate = r.GetMediaItemTakeInfo_Value(take, 'D_PLAYRATE')
  local len = r.GetMediaItemInfo_Value(item, 'D_LENGTH')
  if not src or rate <= 0 or len <= 0.05 then rep.skipped = rep.skipped + 1; return end
  return take, src, rate, r.GetMediaItemTakeInfo_Value(take, 'D_STARTOFFS'), len,
         math.max(1, r.GetMediaSourceNumChannels(src))
end

---------------------------------------------------------------- normalize

-- Phase 1: measure an item. Nothing is changed yet - the gains can only be
-- decided once every selected item has been measured, so that one long track
-- and the same track cut into clips give exactly the same result.
local function analyse_item(item, rep, exec)
  local take, src, rate, offs, len, nch = item_basics(item, rep)
  if not take then return end

  local env, takevol = analyse_raw(item, take, len, nch, exec)
  if not env then rep.failed = rep.failed + 1; return end
  local phrases, stats = find_phrases(env)
  if not phrases or #phrases == 0 then rep.nospeech = rep.nospeech + 1; return end

  local abs_floor = math.min(-50, target - 35)
  local speech = {}
  for _, p in ipairs(phrases) do
    if p.speech then
      local s0 = offs + p.s * rate
      local s1 = offs + p.e * rate
      p.lufs = region_loudness(src, s0, math.min(s1, s0 + (len - p.s) * rate))
      if not p.lufs or p.lufs < abs_floor then p.speech = false
      else speech[#speech + 1] = p end
    end
    coroutine.yield()
  end
  if #speech == 0 then rep.nospeech = rep.nospeech + 1; return end

  return { item = item, take = take, rate = rate, len = len, takevol = takevol,
           pos = r.GetMediaItemInfo_Value(item, 'D_POSITION'),
           phrases = phrases, speech = speech, stats = stats,
           -- the envelope is only needed to place ramps inside pauses
           env = #speech > 1 and env or nil }
end

-- Phase 2: decide every gain, using ONE reference for the whole selection.
-- Evenness is the share of each line's deviation from that reference that gets
-- removed: 100% makes every line identical, lower values leave the performance
-- (a shout stays louder than a whisper) while killing the jumps.
local function shape_gains(all, rep)
  local keep = clamp(1 - evenness / 100, 0, 1)
  local avg = integrated(all, false) or target
  for _, p in ipairs(all) do
    p.gain = (avg + (p.lufs - avg) * keep) - p.lufs
  end

  -- short phrases measure low and sound odd when gained on their own:
  -- keep them near their neighbours (in time, across item borders)
  for i, p in ipairs(all) do
    if p.dur < SHORT_PH then
      local acc, cnt = 0, 0
      for _, j in ipairs({ i - 1, i + 1 }) do
        local q = all[j]
        if q and q.dur >= SHORT_PH then acc = acc + q.gain; cnt = cnt + 1 end
      end
      if cnt > 0 then
        p.gain = clamp(p.gain, acc / cnt - SHORT_DEV, acc / cnt + SHORT_DEV)
      end
    end
  end

  -- anchor: whatever the evenness, the selection as a whole lands on the target
  local off = target - (integrated(all, true) or target)
  for _, p in ipairs(all) do
    p.gain = clamp(p.gain + off, -MAX_CUT, MAX_BOOST)
    local room = CEILING - (p.pk + p.gain)
    if room < 0 then p.gain = p.gain + room; rep.limited = rep.limited + 1 end
    trace('    %8.2fs dur=%4.2f pk=%6.1f lufs=%6.1f -> %+.1f dB (now %.1f)',
      p.t, p.dur, p.pk, p.lufs, p.gain, p.lufs + p.gain)
  end
  trace('  reference %.1f LUFS, evenness %d%% -> target %.1f', avg, evenness, target)
end

-- Phase 3: write the decided gains into the item
local function write_item(rec, rep)
  local item, take, len, rate = rec.item, rec.take, rec.len, rec.rate
  local phrases, speech, env, stats = rec.phrases, rec.speech, rec.env, rec.stats

  -- breaths / noise keep the gain of the nearest phrase, they are never lifted
  for i, p in ipairs(phrases) do
    if not p.speech then
      local prev, nxt
      for j = i - 1, 1, -1 do if phrases[j].speech then prev = phrases[j] break end end
      for j = i + 1, #phrases do if phrases[j].speech then nxt = phrases[j] break end end
      p.gain = (prev or nxt).gain
    end
  end

  local gains = {}
  for _, p in ipairs(speech) do gains[#gains + 1] = p.gain end
  table.sort(gains)
  local base = gains[math.ceil(#gains / 2)]
  local spread = gains[#gains] - gains[1]

  local takemag = math.abs(rec.takevol)
  local takedb = takemag > 1e-6 and lin2db(takemag) or 0
  r.SetMediaItemInfo_Value(item, 'D_VOL', db2lin(base - takedb))

  if #speech == 1 or spread < 0.4 or not env then
    clear_take_env(take)          -- single uniform gain: no envelope needed
  else
    local tenv = ensure_take_vol_env(item, take)
    if tenv then
      local mode = r.GetEnvelopeScalingMode(tenv)
      r.DeleteEnvelopePointRange(tenv, -1e6, 1e6)
      local function put(t, db)
        r.InsertEnvelopePoint(tenv, clamp(t, 0, len) / rate,
          r.ScaleToEnvelopeMode(mode, db2lin(db - base)), 0, 0, false, true)
      end
      if r.CountEnvelopePoints(tenv) > 0 then
        r.SetEnvelopePoint(tenv, 0, 0.0,
          r.ScaleToEnvelopeMode(mode, db2lin(speech[1].gain - base)), 0, 0, false, true)
      else
        put(0, speech[1].gain)
      end
      for i = 1, #speech - 1 do
        local a, b = speech[i], speech[i + 1]
        if math.abs(b.gain - a.gain) > 0.2 then
          local gap = b.s - a.e
          local ramp = math.min(RAMP_MAX, math.max(0.03, gap * 0.45))
          local c = quiet_point(env, a.e, b.s, stats.gate)
          c = clamp(c, a.e + ramp * 0.5, b.s - ramp * 0.5)
          put(c - ramp * 0.5, a.gain)
          put(c + ramp * 0.5, b.gain)
        end
      end
      put(len, speech[#speech].gain)
      r.Envelope_SortPoints(tenv)
      set_env_visible(take, not hide_curves)
    end
  end

  rep.items = rep.items + 1
  rep.phrases = rep.phrases + #speech
  rep.breaths = rep.breaths + (#phrases - #speech)
  rep.min_gain = math.min(rep.min_gain, gains[1])
  rep.max_gain = math.max(rep.max_gain, gains[#gains])
  trace('  item: %d phrases (%d non-speech), gain %+.1f..%+.1f dB, base %+.1f',
    #speech, #phrases - #speech, gains[1], gains[#gains], base)
end

---------------------------------------------------------------- split

local function split_item(item, rep, exec)
  local take, src, rate, offs, len, nch = item_basics(item, rep)
  if not take then return { item } end

  local env = analyse_raw(item, take, len, nch, exec)
  if not env then rep.failed = rep.failed + 1; return { item } end
  local phrases, stats = find_phrases(env)
  if not phrases then rep.nospeech = rep.nospeech + 1; return { item } end
  local regions = speech_regions(phrases, stats, env, len)
  if #regions == 0 then rep.nospeech = rep.nospeech + 1; return { item } end
  if #regions == 1 and regions[1].s < 0.01 and regions[1].e > len - 0.01 then
    rep.items = rep.items + 1
    return { item }                       -- nothing but speech: leave as is
  end
  trace('  split: %d speech region(s) from %d phrases', #regions, #phrases)

  local track = r.GetMediaItem_Track(item)
  local P0 = r.GetMediaItemInfo_Value(item, 'D_POSITION')
  local EPS = 0.005
  local kept, cur = {}, item
  for _, reg in ipairs(regions) do
    if not cur then break end
    local curpos = r.GetMediaItemInfo_Value(cur, 'D_POSITION')
    if P0 + reg.s > curpos + EPS then
      local right = r.SplitMediaItem(cur, P0 + reg.s)
      if right then
        r.DeleteTrackMediaItem(track, cur)   -- silence before the phrase
        rep.removed = rep.removed + 1
        cur = right
      end
    end
    local curend = r.GetMediaItemInfo_Value(cur, 'D_POSITION')
                 + r.GetMediaItemInfo_Value(cur, 'D_LENGTH')
    local piece = cur
    if P0 + reg.e < curend - EPS then
      cur = r.SplitMediaItem(cur, P0 + reg.e) or nil
    else
      cur = nil
    end
    r.SetMediaItemInfo_Value(piece, 'D_FADEINLEN', PAD_LEAD)
    r.SetMediaItemInfo_Value(piece, 'D_FADEOUTLEN', PAD_TRAIL)
    r.SetMediaItemInfo_Value(piece, 'D_SNAPOFFSET', reg.onset - reg.s)
    kept[#kept + 1] = piece
  end
  if cur then
    r.DeleteTrackMediaItem(track, cur)       -- trailing silence
    rep.removed = rep.removed + 1
  end
  rep.items = rep.items + 1
  rep.clips = rep.clips + #kept
  return kept
end

---------------------------------------------------------------- job runner

local job
local ui_status = ''

local function run_job(mode, items, rep)
  local function exec(fn, ...) return coroutine.yield({ fn = fn, args = { ... } }) end

  if mode == 'split' then
    local result = {}
    for i, item in ipairs(items) do
      job.done = i
      if job.cancel then job.items = result; return end
      trace('item %d/%d', i, #items)
      for _, it in ipairs(split_item(item, rep, exec)) do result[#result + 1] = it end
      coroutine.yield()
    end
    job.items = result
    return
  end

  -- measure everything first, so one reference level serves the whole selection
  local recs, all = {}, {}
  for i, item in ipairs(items) do
    job.phase, job.done = 'analysing', i
    if job.cancel then return end
    trace('item %d/%d', i, #items)
    local rec = analyse_item(item, rep, exec)
    if rec then
      recs[#recs + 1] = rec
      for _, p in ipairs(rec.speech) do
        p.t = rec.pos + p.s          -- project time, for ordering across items
        all[#all + 1] = p
      end
    end
    coroutine.yield()
  end
  if #all == 0 then return end
  table.sort(all, function(a, b) return a.t < b.t end)

  shape_gains(all, rep)

  job.total = #recs
  for i, rec in ipairs(recs) do
    job.phase, job.done = 'applying', i
    if job.cancel then return end
    write_item(rec, rep)
    coroutine.yield()
  end
end

local function summarize(mode, rep, cancelled)
  if cancelled then return 'Cancelled - Ctrl+Z undoes partial changes' end
  local extra = {}
  if rep.nospeech > 0 then extra[#extra + 1] = rep.nospeech .. ' no speech' end
  if rep.skipped > 0 then extra[#extra + 1] = rep.skipped .. ' skipped' end
  if rep.failed > 0 then extra[#extra + 1] = rep.failed .. ' unreadable' end
  local msg
  if mode == 'split' then
    if rep.items == 0 then return 'Nothing to split (no speech found)' end
    msg = string.format('%d item(s) -> %d clip(s), %d silent part(s) removed',
      rep.items, rep.clips, rep.removed)
  else
    if rep.items == 0 then return 'Nothing normalized (no speech found)' end
    msg = string.format('%d item(s), %d phrase(s) -> %.1f LUFS   gain %+.1f..%+.1f dB',
      rep.items, rep.phrases, target, rep.min_gain, rep.max_gain)
    if rep.breaths > 0 then extra[#extra + 1] = rep.breaths .. ' breath/noise kept' end
    if rep.limited > 0 then extra[#extra + 1] = rep.limited .. ' peak-limited' end
  end
  if #extra > 0 then msg = msg .. '   (' .. table.concat(extra, ', ') .. ')' end
  return msg
end

local function start_job(mode)
  local items = collect_items()
  if #items == 0 then
    ui_status = 'Select items (or tracks with items) first'
    return
  end
  local rep = { items = 0, phrases = 0, breaths = 0, nospeech = 0, skipped = 0,
                limited = 0, failed = 0, clips = 0, removed = 0,
                min_gain = math.huge, max_gain = -math.huge }
  job = { done = 0, total = #items, cancel = false, rep = rep, mode = mode,
          items = items, phase = 'analysing' }
  job.co = coroutine.create(function() run_job(mode, items, rep) end)
  r.Undo_BeginBlock()
end

local function finish_job()
  local rep, cancelled, mode = job.rep, job.cancel, job.mode
  r.SelectAllMediaItems(0, false)
  for _, it in ipairs(job.items or {}) do
    if r.ValidatePtr2(0, it, 'MediaItem*') then r.SetMediaItemSelected(it, true) end
  end
  r.UpdateArrange()
  r.Undo_EndBlock(mode == 'split' and 'Split dialogue at silence'
                  or string.format('Normalize dialogue to %.1f LUFS', target), -1)
  ui_status = job.error and ('ERROR: ' .. tostring(job.error))
                        or summarize(mode, rep, cancelled)
  trace('%s', ui_status)
  job = nil
end

local function pump_job(budget)
  if not job then return false end
  local t0 = r.time_precise()
  r.PreventUIRefresh(1)
  while true do
    if job.cancel or coroutine.status(job.co) == 'dead' then break end
    local res = job.resume_vals
    job.resume_vals = nil
    local ok, req
    if res then ok, req = coroutine.resume(job.co, table.unpack(res, 1, res.n))
    else ok, req = coroutine.resume(job.co) end
    if not ok then job.error = req; break end
    -- audio APIs fail silently inside coroutines: run them on the main context
    if type(req) == 'table' and req.fn then
      job.resume_vals = table.pack(req.fn(table.unpack(req.args)))
    end
    if r.time_precise() - t0 > budget then break end
  end
  r.PreventUIRefresh(-1)
  if job.cancel or job.error or coroutine.status(job.co) == 'dead' then
    finish_job()
    return false
  end
  return true
end

-- apply the "hide curves" preference to everything currently in scope
local function apply_visibility()
  r.Undo_BeginBlock()
  local n = 0
  for _, item in ipairs(collect_items()) do
    local take = r.GetActiveTake(item)
    if take and not r.TakeIsMIDI(take) then
      set_env_visible(take, not hide_curves)
      n = n + 1
    end
  end
  r.UpdateArrange()
  r.Undo_EndBlock('Toggle take volume curve visibility', -1)
  return n
end

---------------------------------------------------------------- headless

if _G.ALV_HEADLESS then
  if _G.ALV_HIDE_CURVES ~= nil then hide_curves = _G.ALV_HIDE_CURVES end
  if _G.ALV_TARGET   then target = _G.ALV_TARGET end
  if _G.ALV_EVENNESS then evenness = _G.ALV_EVENNESS end
  start_job(_G.ALV_MODE or 'normalize')
  while job do pump_job(3600) end
  return
end

---------------------------------------------------------------- UI

if not r.ImGui_CreateContext then
  start_job('normalize')
  while job do pump_job(3600) end
  r.MB(ui_status, 'Dialogue Normalizer', 0)
  return
end

local ctx = r.ImGui_CreateContext('Dialogue Normalizer')
local ACCENT = 0xE8A33DFF

local function push_style()
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_WindowRounding(), 10)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_FrameRounding(), 6)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_ItemSpacing(), 8, 9)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_WindowPadding(), 18, 16)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_WindowBg(),      0x16171CFF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_TitleBg(),       0x16171CFF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_TitleBgActive(), 0x1E2027FF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_FrameBg(),       0x24262EFF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_FrameBgHovered(),0x2D3039FF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_CheckMark(),     ACCENT)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Button(),        0x2D3039FF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonHovered(), 0x3A3E4AFF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonActive(),  0x464B59FF)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_PlotHistogram(), ACCENT)
end
local function pop_style()
  r.ImGui_PopStyleColor(ctx, 10)
  r.ImGui_PopStyleVar(ctx, 4)
end

local function frame()
  r.ImGui_TextColored(ctx, ACCENT, 'DIALOGUE  NORMALIZER')
  r.ImGui_SameLine(ctx)
  r.ImGui_TextDisabled(ctx, 'EBU R128')
  r.ImGui_Spacing(ctx)

  if not job then
    local rv
    r.ImGui_SetNextItemWidth(ctx, 130)
    rv, target = r.ImGui_InputDouble(ctx, 'Target (LUFS)', target, 0.5, 1.0, '%.1f')
    if rv then
      target = clamp(target, -40, -6)
      r.SetExtState(EXT, 'target', string.format('%.2f', target), true)
    end
    r.ImGui_SetNextItemWidth(ctx, 130)
    rv, evenness = r.ImGui_SliderDouble(ctx, 'Evenness', evenness, 0, 100, '%.0f %%')
    if rv then r.SetExtState(EXT, 'evenness', string.format('%.0f', evenness), true) end
    r.ImGui_TextDisabled(ctx, evenness >= 99 and 'every line hits the target exactly'
      or string.format('keeps %.0f%% of the acting dynamics between lines', 100 - evenness))
    r.ImGui_Spacing(ctx)
  end

  if job then
    r.ImGui_ProgressBar(ctx, job.total > 0 and job.done / job.total or 0, -1, 26,
      string.format('%s  %d / %d',
        job.mode == 'split' and 'splitting' or (job.phase or 'analysing'),
        job.done, job.total))
    r.ImGui_Spacing(ctx)
    if r.ImGui_Button(ctx, 'Cancel', -1, 30) then job.cancel = true end
  else
    r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Button(),        0xD9992BFF)
    r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonHovered(), 0xE8AC45FF)
    r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonActive(),  0xC08820FF)
    r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Text(),          0x16171CFF)
    local go = r.ImGui_Button(ctx, 'NORMALIZE  DIALOGUE', -1, 44)
    r.ImGui_PopStyleColor(ctx, 4)
    if go then ui_status = ''; start_job('normalize') end
    r.ImGui_TextDisabled(ctx, 'selected items, or all items on selected tracks')

    r.ImGui_Spacing(ctx); r.ImGui_Separator(ctx); r.ImGui_Spacing(ctx)

    if r.ImGui_Button(ctx, 'Split at silence', 150, 28) then
      ui_status = ''; start_job('split')
    end
    r.ImGui_SameLine(ctx)
    r.ImGui_TextDisabled(ctx, 'cut into clips first, then normalize')

    local rv, val = r.ImGui_Checkbox(ctx, 'Hide volume curves', hide_curves)
    if rv then
      hide_curves = val
      r.SetExtState(EXT, 'hide_curves', hide_curves and '1' or '0', true)
      local n = apply_visibility()
      ui_status = string.format('%s volume curves on %d item(s)',
        hide_curves and 'Hid' or 'Showed', n)
    end
  end

  if ui_status ~= '' then
    r.ImGui_Spacing(ctx); r.ImGui_Separator(ctx); r.ImGui_Spacing(ctx)
    local col = ui_status:find('^ERROR') and 0xE06868FF
             or ui_status:find('^Cancelled') and 0xE0C468FF
             or ui_status:find('^Select') and 0xE0C468FF or 0x8FCF8FFF
    r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Text(), col)
    r.ImGui_TextWrapped(ctx, ui_status)
    r.ImGui_PopStyleColor(ctx, 1)
  end
end

local loop
loop = function()
  local win_open
  local okf, err = pcall(function()
    if job then pump_job(0.05) end
    push_style()
    r.ImGui_SetNextWindowSize(ctx, 430, 370, r.ImGui_Cond_FirstUseEver())
    local visible, open = r.ImGui_Begin(ctx, 'Dialogue Normalizer', true)
    if visible then frame(); r.ImGui_End(ctx) end
    pop_style()
    win_open = open
  end)
  if not okf then
    r.MB('Dialogue Normalizer UI error:\n' .. tostring(err), 'Dialogue Normalizer', 0)
    return
  end
  if win_open then r.defer(loop)
  elseif job then job.cancel = true; pump_job(3600) end
end

r.defer(loop)
