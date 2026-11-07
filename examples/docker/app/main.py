"""Runs in the final image, which has no Rust: the prebuilt libraries in __pyrust__/ are used."""
from kernels import checksum, word_counts

print(word_counts('the cat and the hat'))
print(checksum(1_000_000))
print('backend:', checksum.report()['backend'], '| loaded from prebuilt:', checksum.report()['cache_hit'])
