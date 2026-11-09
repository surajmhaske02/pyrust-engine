"""Value kinds shared by the compiler, the string/container code generator and the runtime.

Scalars: 'i64' int, 'f64' float, 'f32' NumPy float32, 'bool', 'char' (one character),
'strv' (a string value). Sequences passed as (pointer, rows, cols): NumPy arrays, lists
of numbers, pandas and polars Series, str and bytes (see ARRAY_INFO). Series of str
('sseries' pandas, 'plsseries' polars) are passed like lists of str. Containers: 'slist' (list of
str), 'dict(K,V)' and 'set(K)' with K in i64/strv and V in i64/f64/bool/strv.
"""

# Every sequence kind is passed to Rust as (pointer, rows, cols) and read through Arr<T>.
ARRAY_INFO = {
    'array': ('f64', 1),
    'iarray': ('i64', 1),
    'array2': ('f64', 2),
    'iarray2': ('i64', 2),
    'farray': ('f32', 1),
    'farray2': ('f32', 2),
    'flist': ('f64', 1),
    'ilist': ('i64', 1),
    'series': ('f64', 1),
    'iseries': ('i64', 1),
    'fseries': ('f32', 1),
    'plseries': ('f64', 1),
    'iplseries': ('i64', 1),
    'str': ('u32', 1),
    'bytes': ('u8', 1),
    # NumPy int32 / uint8 / bool arrays (elements: see SMALL_READ)
    'i32array': ('i32', 1),
    'i32array2': ('i32', 2),
    'u8array': ('u8', 1),
    'u8array2': ('u8', 2),
    'barray': ('bool', 1),
    'barray2': ('bool', 2),
    # Read-only views of NumPy arrays: a[lo:hi] and rows m[i] / m[i, :]
    'aview': ('f64', 1),
    'iaview': ('i64', 1),
    'faview': ('f32', 1),
}
ARRAY_KINDS = tuple(ARRAY_INFO)
NUMPY_KINDS = (
    'array',
    'iarray',
    'array2',
    'iarray2',
    'farray',
    'farray2',
    'i32array',
    'i32array2',
    'u8array',
    'u8array2',
    'barray',
    'barray2',
)
VIEW_KINDS = ('aview', 'iaview', 'faview')
VIEW_OF = {
    'array': 'aview',
    'array2': 'aview',
    'aview': 'aview',
    'iarray': 'iaview',
    'iarray2': 'iaview',
    'iaview': 'iaview',
    'farray': 'faview',
    'farray2': 'faview',
    'faview': 'faview',
}
# Elements of int32 / uint8 / bool arrays are NumPy scalars that wrap around on overflow, so
# compiled code only compares, tests, converts (int(), float(), bool()) and stores them.
SMALL_READ = {
    'i32array': 'npi32',
    'i32array2': 'npi32',
    'u8array': 'npu8',
    'u8array2': 'npu8',
    'barray': 'npbool',
    'barray2': 'npbool',
}
NP_SMALL = ('npi32', 'npu8', 'npbool')
LIST_KINDS = ('flist', 'ilist')
READ_ONLY_KINDS = ('series', 'iseries', 'fseries', 'plseries', 'iplseries', 'str', 'bytes', *VIEW_KINDS)
# pandas Series: s[i] looks up the label i, so negative i raises KeyError (polars indexes like a list).
PANDAS_SERIES = ('series', 'iseries', 'fseries')
SERIES_NAMES = {
    'series': 'pandas Series',
    'iseries': 'pandas Series',
    'fseries': 'pandas Series',
    'plseries': 'polars Series',
    'iplseries': 'polars Series',
    'str': 'str',
    'bytes': 'bytes',
    'aview': 'NumPy slice (a read-only view)',
    'iaview': 'NumPy slice (a read-only view)',
    'faview': 'NumPy slice (a read-only view)',
}
# Series of str are compiled as read-only lists of str with Series indexing rules.
STR_SERIES = ('sseries', 'plsseries')
ELEMENT = {k: v[0] for k, v in ARRAY_INFO.items()}
# What reading one element gives: str elements are one-character strings, bytes elements are ints.
READ_KIND = {'f64': 'f64', 'i64': 'i64', 'f32': 'f32', 'u32': 'char', 'u8': 'i64', 'i32': 'npi32', 'bool': 'npbool'}
SCALARS = ('i64', 'f64', 'bool')
VALUES = ('i64', 'f64', 'bool', 'char', 'f32', 'strv', 'npi32', 'npu8', 'npbool')  # values a variable can hold
TEXT = ('str', 'strv', 'char')
RUST_TYPE = {
    'i64': 'i64',
    'f64': 'f64',
    'f32': 'f32',
    'bool': 'bool',
    'char': 'u32',
    'strv': 'S',
    'none': '()',
    'npi32': 'i32',
    'npu8': 'u8',
    'npbool': 'bool',
}
ZERO = {
    'i64': '0',
    'f64': '0.0',
    'f32': '0.0',
    'bool': 'false',
    'char': '0',
    'strv': 'S::default()',
    'npi32': '0',
    'npu8': '0',
    'npbool': 'false',
}
KEY_KINDS = ('i64', 'strv')
ITEM_KINDS = ('i64', 'f64', 'bool', 'strv')
# Kinds served by the pyrust_fast entry point (value and status returned together).
FAST_KINDS = ('i64', 'f64', 'bool')
FAST_RETURNS = ('i64', 'f64', 'bool', 'none')
# Kinds the direct CPython entry point reads itself (numbers, NumPy arrays, str, bytes).
DIRECT_KINDS = (
    'i64',
    'f64',
    'bool',
    'array',
    'iarray',
    'array2',
    'iarray2',
    'farray',
    'farray2',
    'i32array',
    'i32array2',
    'u8array',
    'u8array2',
    'barray',
    'barray2',
    'str',
    'bytes',
    'flist',
    'ilist',
)
PLACEHOLDERS = {'list?': 'flist', 'dict?': 'dict(i64,i64)', 'set?': 'set(i64)'}


# Classes of objects whose methods are compiled, by token (the token is part of the kind).
CLASSES: dict = {}


def class_token(cls):
    token = f'{id(cls):x}'
    CLASSES[token] = cls
    return token


def object_kind(parts, cls=None):
    """'obj(attr:kind,...)' for an object whose attributes the function uses; an assigned
    attribute is marked with '!'. With a class, 'obj@token(...)': its methods are compiled too."""
    tag = f'@{class_token(cls)}' if cls is not None else ''
    return f'obj{tag}(' + ','.join(f'{a}:{k}{"!" if w else ""}' for a, k, w in parts) + ')'


def object_parts(kind):
    """[(attr, kind, assigned)] of an object kind, or None for other kinds."""
    if not (isinstance(kind, str) and kind.startswith('obj') and kind.endswith(')') and '(' in kind):
        return None
    body = kind[kind.index('(') + 1 : -1]
    out = []
    for part in _top_level(body):
        attr, item = part.split(':', 1)
        out.append((attr, item.rstrip('!'), item.endswith('!')))
    return out


def _top_level(text):
    """Split at commas that are not inside parentheses (attribute kinds like dict(strv,f64))."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    if text:
        parts.append(text[start:])
    return parts


def object_class(kind):
    """The class of an 'obj@token(...)' kind, or None."""
    if not (isinstance(kind, str) and kind.startswith('obj@')):
        return None
    return CLASSES.get(kind[4 : kind.index('(')])


def tuple_kind(kinds):
    return 'tuple(' + ','.join(kinds) + ')'


def tuple_parts(kind):
    return kind[6:-1].split(',') if kind.startswith('tuple(') else None


def is_dict(kind):
    return isinstance(kind, str) and kind.startswith('dict(')


def is_set(kind):
    return isinstance(kind, str) and kind.startswith('set(')


def dict_parts(kind):
    return kind[5:-1].split(',')


def set_part(kind):
    return kind[4:-1]


def is_container(kind):
    """Lists of strings, dicts and sets: passed by reference, never aliased."""
    return kind == 'slist' or is_dict(kind) or is_set(kind) or kind in PLACEHOLDERS


def is_packed(kind):
    """Kinds exchanged with Python as one packed byte buffer."""
    return kind in ('strv', 'slist') or is_dict(kind) or is_set(kind)


def resolve(kind):
    return PLACEHOLDERS.get(kind, kind)


def rust_type(kind):
    kind = resolve(kind)
    if kind in ARRAY_INFO:
        return f'Arr<{ARRAY_INFO[kind][0]}>'
    if kind == 'slist':
        return 'Vec<S>'
    if is_dict(kind):
        k, v = dict_parts(kind)
        return f'OMap<{RUST_TYPE[k]},{RUST_TYPE[v]}>'
    if is_set(kind):
        return f'HSet<{RUST_TYPE[set_part(kind)]}>'
    parts = tuple_parts(kind)
    if parts:
        return '(' + ','.join(RUST_TYPE[p] for p in parts) + ')'
    return RUST_TYPE[kind]


def zero(kind):
    """Rust initial value for a local of this kind."""
    kind = resolve(kind)
    if kind in ARRAY_INFO:
        return 'Arr::empty()'
    if kind == 'slist':
        return 'Vec::new()'
    if is_dict(kind):
        return 'OMap::new()'
    if is_set(kind):
        return 'HSet::default()'
    return ZERO[kind]
