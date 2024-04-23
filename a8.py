from neural import *

# print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

# xor_data = [
#     ([0,0], [0]),
#     ([1,0], [1]),
#     ([0,1], [1]),
#     ([0,0], [0])
# ]

# xor_nn = NeuralNet(2, 2, 1)
# xor_nn.train(xor_data, iters=100, print_interval=10)

# print()

# xor_nn1 = NeuralNet(2, 2, 1)
# xor_nn1.train(xor_data, iters=100, print_interval=10)

# print()

# xor_nn2 = NeuralNet(2, 2, 1)
# xor_nn2.train(xor_data, iters=1000, print_interval=100)


# print("<<<2>>>")

# xor_nn = NeuralNet(2, 8, 1)
# xor_nn.train(xor_data, iters=1000, print_interval=100)

# print("<<<3>>>")

# xor_nn = NeuralNet(2, 1, 1)
# xor_nn.train(xor_data, iters=1000, print_interval=100)

# # print(xor_nn.get_ih_weights())
# # print()
# # print(xor_nn.get_ho_weights())

# # print(xor_nn.test_with_expected(xor_data))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

voter_data = [
    ([.9, .6, .8, .3, .1], [1.0]),
    ([.8, .8, .4, .6, .4], [1.0]),
    ([.7, .2, .4, .6, .3], [1.0]),
    ([.5, .5, .8, .4, .8], [0.0]),
    ([.3, .1, .6, .8, .8], [0.0]),
    ([.6, .3, .4, .3, .6], [0.0])
]

voter_nn = NeuralNet(5, 1, 1)
voter_nn.train(voter_data)
# print(voter_nn.test_with_expected(voter_data))

print()

print("Finding Affiliated Party:")

print()

voter_test = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]

print(voter_nn.test(voter_test))