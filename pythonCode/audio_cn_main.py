from tfl_audio_convnets import *
def validate3(ann):
    print("validating all three")
    validA = validate_tfl_audio_convnet_model(ann,
                                                BUZZ1_valid_X,
                                                BUZZ1_valid_Y) 
    print(f'BUZZ1: {validA}')
    validB = validate_tfl_audio_convnet_model(ann,
                                                BUZZ2_valid_X,
                                                BUZZ2_valid_Y) 
    print(f'BUZZ2: {validB}')
    validC = validate_tfl_audio_convnet_model(ann,
                                                BUZZ3_valid_X,
                                                BUZZ3_valid_Y)
    print(f'BUZZ4: {validC}')
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
    big_data_train_Y = np.append(big_data_train_Y, BUZZ3_train_Y, axis=0)
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
# ann = make_200X10_relu_1conv(learn_rate=0.1)
# ann1 = load_256X10_relu_1conv(NETPATH + '256X10_1conv.tfl', learn_rate=0.02)
# train_tfl_audio_convnet_model(ann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=4, batch_size=15)
# validate3(ann)
# ann.save(NETPATH + '200X10_relu_1conv.tfl')
# train_tfl_audio_convnet_model(ann1, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=20, batch_size=20)
# validate3(ann1)
# ann1.save(NETPATH + '256X10_1conv_bigtrain.tfl')
# ann = load_256X10_relu_1conv(NETPATH + '256X10_1conv_bigtrain.tfl')
# ann = load_256X10_relu_1conv(NETPATH + '256X10_1conv_bigtrain.tfl', learn_rate=0.02)
# ann = make_32_relu_1conv(learn_rate=0.1)
# ann = make_256X10_relu_1conv_bigFilt(learn_rate=0.1)
# ann = load_256X10_relu_1conv_bigFilt(NETPATH + '256X10_relu_1conv_bigFilt_v1.tfl', learn_rate=0.04)
# ann = make_256X10_relu_1conv_bigFilt(learn_rate=0.1)
# ann = make_100X20X3_1conv_bigfilt(learn_rate=0.1)
# ann = load_100X20X2_relu_1conv_bigfilt(NETPATH + '100X20_relu_1conv_.tfl', learn_rate=0.05)
ann = load_256X10_relu_1conv(NETPATH + '256X10_1conv_bigtrain.tfl')
# train_tfl_audio_convnet_model(ann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=5, batch_size=20)
validate3(ann)
ann.save(NETPATH + 'aud_cn.tfl')
