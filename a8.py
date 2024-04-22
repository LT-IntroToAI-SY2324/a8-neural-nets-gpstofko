from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_data = [
    ([0,0], [0]),
    ([1,0], [1]),
    ([0,1], [1]),
    ([0,0], [0])
]

xor_nn = NeuralNet(2, 2, 1)
xor_nn.train(xor_data, iters=100, print_interval=10)

print()

xor_nn1 = NeuralNet(2, 2, 1)
xor_nn1.train(xor_data, iters=100, print_interval=10)

print()

xor_nn2 = NeuralNet(2, 2, 1)
xor_nn2.train(xor_data, iters=1000, print_interval=100)


print("<<<2>>>")

xor_nn = NeuralNet(2, 8, 1)
xor_nn.train(xor_data, iters=1000, print_interval=100)

print("<<<3>>>")

xor_nn = NeuralNet(2, 1, 1)
xor_nn.train(xor_data, iters=1000, print_interval=100)

# print(xor_nn.get_ih_weights())
# print()
# print(xor_nn.get_ho_weights())

# print(xor_nn.test_with_expected(xor_data))



