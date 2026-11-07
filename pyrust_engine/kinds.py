"""Value kinds shared by the compiler, the string/container code generator and the runtime.

Scalars: 'i64' int, 'f64' float, 'f32' NumPy float32, 'bool', 'char' (one character),
'strv' (a string value). Sequences passed as (pointer, rows, cols): NumPy arrays, lists
of numbers, pandas Series, str and bytes (see ARRAY_INFO). Containers: 'slist' (list of
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
    'str': ('u32', 1),
    'bytes': ('u8', 1),
}
ARRAY_KINDS = tuple(ARRAY_INFO)
NUMPY_KINDS = ('array', 'iarray', 'array2', 'iarray2', 'farray', 'farray2')
LIST_KINDS = ('flist', 'ilist')
READ_ONLY_KINDS = ('series', 'iseries', 'str', 'bytes')
ELEMENT = {k: v[0] for k, v in ARRAY_INFO.items()}
# What reading one element gives: str elements are one-character strings, bytes elements are ints.
READ_KIND = {'f64': 'f64', 'i64': 'i64', 'f32': 'f32', 'u32': 'char', 'u8': 'i64'}
SCALARS = ('i64', 'f64', 'bool')
VALUES = ('i64', 'f64', 'bool', 'char', 'f32', 'strv')  # values a variable can hold
TEXT = ('str', 'strv', 'char')
RUST_TYPE = {'i64': 'i64', 'f64': 'f64', 'f32': 'f32', 'bool': 'bool', 'char': 'u32', 'strv': 'S', 'none': '()'}
ZERO = {'i64': '0', 'f64': '0.0', 'f32': '0.0', 'bool': 'false', 'char': '0', 'strv': 'S::default()'}
KEY_KINDS = ('i64', 'strv')
ITEM_KINDS = ('i64', 'f64', 'bool', 'strv')
PLACEHOLDERS = {'list?': 'flist', 'dict?': 'dict(i64,i64)', 'set?': 'set(i64)'}


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
        return f'HashSet<{RUST_TYPE[set_part(kind)]}>'
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
        return 'HashSet::new()'
    return ZERO[kind]
