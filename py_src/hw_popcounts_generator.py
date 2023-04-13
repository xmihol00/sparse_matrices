from quine_mccluskey.qm import QuineMcCluskey

simplyfier = QuineMcCluskey(True)

print(simplyfier.simplify([1, 2, 4, 7, 8, 11, 13, 14], num_bits=4))
print(simplyfier.simplify([3, 5, 6, 7, 9, 10, 11, 12, 13, 14], num_bits=4))
print(simplyfier.simplify([15], num_bits=4))

#{'^^^^'}
#{'1-^^', '-^^1', '^1^-'}
#{'1111'}