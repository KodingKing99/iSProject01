# from pythonCode.tfl_image_convnets import train_tfl_image_convnet_model
from tfl_audio_anns import *
def validate3(ann):
    print("validating all three")
    validA = validate_tfl_audio_ann_model(ann,
                                                BUZZ1_valid_X,
                                                BUZZ1_valid_Y) 
    print(f'BEE1: {validA}')
    validB = validate_tfl_audio_ann_model(ann,
                                                BUZZ2_valid_X,
                                                BUZZ2_valid_Y) 
    print(f'BEE2: {validB}')
    validC = validate_tfl_audio_ann_model(ann,
                                                BUZZ3_valid_X,
                                                BUZZ3_valid_Y)
    print(f'BEE4: {validC}')
def randomizeData(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b
def makeBigData():
    print("making big data...")
    print("making big_data_train_X...")
    big_data_train_X = BUZZ1_train_X
    big_data_train_X = np.append(big_data_train_X, BUZZ2_train_X, axis=0)
    big_data_train_X = np.append(big_data_train_X, BUZZ3_train_X, axis=0)
    print("DONE!")
    print("making big_data_train_Y...")
    big_data_train_Y = BUZZ1_train_Y 
    big_data_train_Y = np.append(big_data_train_Y, BUZZ2_train_Y, axis=0)
    big_data_train_Y = np.append(big_data_train_Y, BUZZ3_train_Y , axis=0)
    big_data_train_X, big_data_train_Y = randomizeData(big_data_train_X, big_data_train_Y)
    print("DONE!")
    print("making big_data_test_X...")
    big_data_test_X = BUZZ1_test_X 
    # big_data_test_X = BEE2_1S_test_X
    big_data_test_X = np.append(big_data_test_X, BUZZ2_test_X, axis=0)
    big_data_test_X = np.append(big_data_test_X, BUZZ3_test_X, axis=0)
    print("DONE!")
    print("making big_data_test_Y...")
    big_data_test_Y = BUZZ1_test_Y
    # big_data_test_Y = BEE2_1S_test_Y
    big_data_test_Y = np.append(big_data_test_Y, BUZZ2_test_Y, axis=0)
    big_data_test_Y = np.append(big_data_test_Y, BUZZ3_test_Y, axis=0)
    print("DONE!")
    return big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y
big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y = makeBigData()
print(big_data_train_X.shape)
print(big_data_train_Y.shape)
print(big_data_test_X.shape)
print(big_data_test_Y.shape)
# ann = load_256X10_relu(NETPATH + '256X10_relu.tfl')
# print("256X10_relu")
# train_tfl_audio_ann_model(ann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=25, batch_size=15)
# validate3(ann)
# ann.save(NETPATH + '256X10_relu_bigboy.tfl')
# ann = load_256X10_relu(NETPATH + '256X10_relu.tfl')
# ann = make_500_relu_X10_softmax()
# ann = load_500_relu_X10_softmax(NETPATH + '500X10_relu.tfl')
# print("500X10_relu")
# train_tfl_audio_ann_model(ann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=25, batch_size=15)
# validate3(ann)
# ann.save(NETPATH + '500X10_relu_bigboy.tfl')
# ann = make_200_relu_X10_softmax() 
# ann = load_200_relu_X10_softmax(NETPATH + "200X10_tanh.tfl")
# ann = make_200_relu_X10_softmax()
# ann = load_200_relu_X10_softmax(NETPATH + '200X10_relu_bigBoy.tfl', learn_rate=0.01)
# ann = load_256X10_relu(NETPATH + '256X')
# ann = make_256X10_relu(learn_rate=0.04)
ann = load_256X10_relu(NETPATH + '256X10_relu_bigboy.tfl', learn_rate=0.02)
print(ann)
# ann = load_500_relu_X10_softmax(NETPATH + '500X10_relu.tfl')
# print("200X10_tanh_relu")
train_tfl_audio_ann_model(ann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=50, batch_size=15)
validate3(ann)
ann.save(NETPATH + '256X10_relu_bigboy_v2.tfl')