"""Exchange strings, lists of strings, dicts and sets with native code as one byte buffer.

Layout (little-endian 64-bit words; strings as UTF-32 code points):
  column of ints / floats / bools: one word per item
  column of strings: a separator code point and the total, then either the strings joined
    by the separator (from native code: Python splits them in C) or, with separator
    2**64-1, one length per string and all characters
  list or set: item count, then one column;  dict: count, key column, value column
  single string: a column with one string
Built with C-speed helpers (array, str.join, bytes.decode), not per-item Python code.
"""

import struct
from array import array
from itertools import accumulate

_COUNT = struct.Struct('<Q')
_NO_SEPARATOR = 2**64 - 1


def _put_column(kind, values, out):
    if kind == 'strv':
        lengths = array('q', map(len, values))
        out.append(_COUNT.pack(_NO_SEPARATOR))  # lengths follow (no separator)
        out.append(_COUNT.pack(sum(lengths)))
        out.append(lengths.tobytes())
        out.append(''.join(values).encode('utf-32-le', 'surrogatepass'))
    elif kind == 'f64':
        out.append(array('d', values).tobytes())
    else:  # i64, bool
        out.append(array('q', values).tobytes())


def pack(kind, value):
    """Bytes for a list of str ('slist'), 'set(K)' or 'dict(K,V)' argument."""
    out = []
    if kind == 'slist':
        out.append(_COUNT.pack(len(value)))
        _put_column('strv', value, out)
    elif kind.startswith('set('):
        items = list(value)
        out.append(_COUNT.pack(len(items)))
        _put_column(kind[4:-1], items, out)
    else:
        key, item = kind[5:-1].split(',')
        out.append(_COUNT.pack(len(value)))
        _put_column(key, list(value.keys()), out)
        _put_column(item, list(value.values()), out)
    return b''.join(out)


def _get_column(kind, data, pos, n):
    if kind == 'strv':
        separator = _COUNT.unpack_from(data, pos)[0]
        total = _COUNT.unpack_from(data, pos + 8)[0]
        pos += 16
        if separator != _NO_SEPARATOR:
            # Native code joined the strings with a code point none of them contains.
            text = bytes(data[pos : pos + 4 * total]).decode('utf-32-le', 'surrogatepass')
            return (text.split(chr(separator)) if n else []), pos + 4 * total
        lengths = array('q')
        lengths.frombytes(data[pos : pos + 8 * n])
        pos += 8 * n
        text = bytes(data[pos : pos + 4 * total]).decode('utf-32-le', 'surrogatepass')
        pos += 4 * total
        ends = list(accumulate(lengths))
        starts = [0, *ends[:-1]]
        return [text[a:b] for a, b in zip(starts, ends)], pos
    values = array('d' if kind == 'f64' else 'q')
    values.frombytes(data[pos : pos + 8 * n])
    pos += 8 * n
    items = values.tolist()
    if kind == 'bool':
        items = [x != 0 for x in items]
    return items, pos


def unpack(kind, data):
    """Python value for a packed 'strv', 'slist', 'set(K)' or 'dict(K,V)' result."""
    data = memoryview(data)
    if kind == 'strv':
        return _get_column('strv', data, 0, 1)[0][0]
    n = _COUNT.unpack_from(data, 0)[0]
    if kind == 'slist':
        return _get_column('strv', data, 8, n)[0]
    if kind.startswith('set('):
        return set(_get_column(kind[4:-1], data, 8, n)[0])
    key, item = kind[5:-1].split(',')
    keys, pos = _get_column(key, data, 8, n)
    values, _ = _get_column(item, data, pos, n)
    return dict(zip(keys, values))
