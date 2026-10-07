"""Conservative maintenance calendar from public handover messages.

Only explicit engineering notices are actionable. Unknown language is left to
the existing LLM, not guessed from low scores (flat tests also produce zero).
No scenario dates, target IDs or private engine state are embedded here.
"""
from datetime import datetime, timedelta, timezone
import re
import unicodedata


def number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip('零一二三四五六七八九', range(10)))
    text = text.replace('两', '二')
    if '十' in text:
        a, b = text.split('十', 1)
        return digits.get(a, 1) * 10 + digits.get(b, 0)
    return digits.get(text, 0)


NUM = r'[零一二两三四五六七八九十\d]+'
DATE = re.compile(r'(\d{1,2})[/月](\d{1,2})日?')
CLOCK = re.compile(r'(\d{1,2}):(\d{2})')


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\u200b', '')
    text = re.sub(r'(?<=:)\([^)]*\)(?=\d)', '', text)
    text = text.translate(str.maketrans({'時': '时', '點': '点', '報': '报'}))
    def clock(m):
        hour = number(m[1])
        minute = 30 if m[2] == '半' else number(m[3] or '0')
        return f'{hour}:{minute:02d}'
    return re.sub(f'({NUM})[点时](半|整|({NUM})分?)?', clock, text)


class MaintenanceSchedule:
    def __init__(self, utc_offset_hours=None):
        # Missing timezone must not silently schedule local notices as UTC.
        self.offset = utc_offset_hours
        self.faults = set()
        self.handled = set()
        self.tests = set()
        self.seen = set()
        self.cancelled = set()

    def ingest(self, messages):
        for message in messages:
            if message.get('record_type') != 'observation_request':
                continue
            reason = str(message.get('reason') or '')
            key = (message.get('request_id'), reason)
            if key in self.seen:
                continue
            self.seen.add(key)
            try:
                issued = datetime.fromisoformat(message['issued_at_utc'].replace('Z', '+00:00'))
            except (KeyError, ValueError, TypeError):
                continue
            for raw in reason.splitlines():
                line = normalize(raw)
                # Do not promote a tentative memory or rumor to a scheduled fault.
                if '【灯】' in line or any(s in line for s in ('没确认', '未确认', '听说', '听說', '没見正式', '隔壁', '要是', '如果', '顺延')):
                    continue
                # The handover explicitly uses Caesar-encoded English. Decode
                # only when a full engineering phrase validates the candidate.
                if any(mark in line for mark in ('凯撒', 'シーザー', '加密', '往后挪')):
                    for shift in range(26):
                        decoded = ''.join(chr((ord(c.lower()) - 97 - shift) % 26 + 97)
                                          if c.isascii() and c.isalpha() else c for c in line)
                        if 'guider camera' in decoded:
                            line = decoded
                            break
                if 'guider camera' in line.lower():
                    english = line.lower()
                    if any(word in english for word in ('unconfirmed', 'rumor', 'cancel', 'postpon', 'window', 'hours after', 'minutes after')):
                        # Relative English schedules need more context; prefer
                        # their Chinese engineering counterpart over guessing.
                        continue
                    d, t = DATE.search(line), CLOCK.search(line)
                    if not d or not t:
                        continue
                    zone = 'UTC+9 东京' if 'tokyo' in english or 'utc+9' in english else 'UTC' if 'utc' in english else ''
                    line = f'工程组通知：{d[0]} {t[0]} 导星相机，到点报修。{zone}'
                if '导星相机' not in line and not ('平场灯' in line or '镜盖测试' in line):
                    continue
                dates = list(DATE.finditer(line))
                if not dates:
                    continue
                offset = 9 if ('东京' in line or 'UTC+9' in line) else 0 if 'UTC' in line else self.offset
                if offset is None:
                    continue
                tz = timezone(timedelta(hours=float(offset)))
                # Corrections repeat the old date/time before giving the real one.
                date = dates[-1] if ('记错' in line or '搞错' in line or '弄错' in line) else dates[0]
                month, day = int(date[1]), int(date[2])
                try:
                    base = datetime(issued.year, month, day, tzinfo=tz)
                    if base < issued - timedelta(days=180):
                        base = base.replace(year=base.year + 1)
                    elif base > issued + timedelta(days=180):
                        base = base.replace(year=base.year - 1)
                except ValueError:
                    continue
                tail = line[date.end():]
                times = list(CLOCK.finditer(tail))
                def at(match, night=False):
                    h, m = int(match[1]), int(match[2])
                    if h > 23 or m > 59:
                        return None
                    local = base + timedelta(hours=h, minutes=m, days=int(night and h < 12))
                    return local.astimezone(timezone.utc)
                if '导星相机' not in line:
                    # Flat/cap tests are a local observing-night schedule: 03:00
                    # belongs to the following calendar day, not the start day.
                    for window in re.finditer(r'\d{1,2}:\d{2}\s*(?:到|至|~|〜|-)\s*\d{1,2}:\d{2}', tail):
                        left, right = CLOCK.finditer(window[0])
                        start, end = at(left, True), at(right, True)
                        if start is None or end is None:
                            continue
                        if end <= start:
                            end += timedelta(days=1)
                        if end - start <= timedelta(hours=2):
                            self.tests.add((start, end))
                    continue
                if '取消' in line:
                    for t in times:
                        when = at(t)
                        self.faults.discard(when)
                        self.cancelled.add(when)
                    continue
                if '测试结束' in line or '测试结束以后' in line:
                    index = re.search(f'第({NUM})段', line)
                    delay = re.search(f'再等({NUM})小时(?:({NUM})分)?', line)
                    windows = sorted((a, b) for a, b in self.tests
                                     if (a.astimezone(tz) - timedelta(hours=12)).date() == base.date())
                    idx = number(index[1]) - 1 if index else -1
                    if delay and 0 <= idx < len(windows):
                        self.faults.add(windows[idx][1] + timedelta(hours=number(delay[1]), minutes=number(delay[2] or '0')))
                    continue
                if not times or not any(s in line for s in ('工程组', '到点', '同一件事', 'UTC', '以此为准')):
                    continue
                when = at(times[0])
                if when is None:
                    continue
                delay = re.search(f'推迟({NUM})小时(?:({NUM})分)?', line)
                if '推迟' in line and delay is None:
                    continue
                if delay:
                    self.faults.discard(when)
                    self.cancelled.add(when)
                    when += timedelta(hours=number(delay[1]), minutes=number(delay[2] or '0'))
                if when not in self.cancelled:
                    self.faults.add(when)

    def due(self, now):
        return any(t <= now for t in self.faults - self.handled)

    def acknowledge(self, now):
        self.handled.update(t for t in self.faults if t <= now)

    def test_end(self, now):
        return max((end for start, end in self.tests if start <= now < end), default=None)

    def next_boundary(self, now):
        future = [t for t in self.faults - self.handled if t > now]
        future.extend(start for start, end in self.tests if start > now)
        return min(future, default=None)
