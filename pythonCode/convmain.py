from tfl_image_convnets import *
def validate3(ann):
    print("validating all three")
    validA = validate_tfl_image_convnet_model(ann,
                                                BEE1_valid_X,
                                                BEE1_valid_Y) 
    print(f'BEE1: {validA}')
    validB = validate_tfl_image_convnet_model(ann,
                                                BEE2_1S_valid_X,
                                                BEE2_1S_valid_Y) 
    print(f'BEE2: {validB}')
    validC = validate_tfl_image_convnet_model(ann,
                                                BEE4_valid_X,
                                                BEE4_valid_Y)
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
    # big_data_train_X = BEE1_train_X[:5000]
    big_data_train_X = BEE4_train_X
    big_data_train_X = np.append(big_data_train_X, BEE2_1S_train_X[:10000], axis=0)
    big_data_train_X = np.append(big_data_train_X, BEE1_train_X[:5000], axis=0)
    # np.random.shuffle(big_data_train_X)
    print("DONE!")
    print("making big_data_train_Y...")
    # big_data_train_Y = BEE1_train_Y[:5000]
    big_data_train_Y = BEE4_train_Y
    big_data_train_Y = np.append(big_data_train_Y, BEE2_1S_train_Y[:10000], axis=0)
    big_data_train_Y = np.append(big_data_train_Y, BEE1_train_Y[:5000] , axis=0)
    big_data_train_X, big_data_train_Y = randomizeData(big_data_train_X, big_data_train_Y)
    # np.random.shuffle(big_data_train_Y)
    print("DONE!")
    print("making big_data_test_X...")
    big_data_test_X = BEE1_test_X
    # big_data_test_X = BEE2_1S_test_X
    big_data_test_X = np.append(big_data_test_X, BEE2_1S_test_X, axis=0)
    big_data_test_X = np.append(big_data_test_X, BEE4_test_X, axis=0)
    print("DONE!")
    print("making big_data_test_Y...")
    big_data_test_Y = BEE1_test_Y
    # big_data_test_Y = BEE2_1S_test_Y
    big_data_test_Y = np.append(big_data_test_Y, BEE2_1S_test_Y, axis=0)
    big_data_test_Y = np.append(big_data_test_Y, BEE4_test_Y, axis=0)
    print("DONE!")
    return big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y
big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y = makeBigData()
print(big_data_train_X.shape)
print(big_data_train_Y.shape)
print(big_data_test_X.shape)
print(big_data_test_Y.shape)
# ann = make_image_convnet_model(learn_rate=0.05)
# train_tfl_image_convnet_model(ann, big_data_train_X, big_data_test_Y, big_data_test_X, big_data_test_Y, num_epochs=4)
# validate3(ann)
# ann.save(NETPATH + 'vlad_convnet_v2.cn')
ann1 = load_1conv_256X10_relu_model(NETPATH + 'image_cn.tfl', learn_rate=0.05)
# ann1 = make_1conv_256X10_relu_model()
# print(ann1)
# ann2 = make_1conv_40Filter_256X10_relu_model()
# print(ann2)
# ann3 = make_2conv_40X10Filter_256X10_relu_model()
# print(ann3)
# train_tfl_image_convnet_model(ann1, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=10, batch_size=20)
validate3(ann1)
ann1.save(SAVEPATH + 'image_cn.tfl')
# train_tfl_image_convnet_model(ann2, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=4)
# validate3(ann2)
# ann2.save(NETPATH + '1conv_40Filter_256X10_relu.cn')
# train_tfl_image_convnet_model(ann3, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=4)
# validate3(ann3)
# ann3.save(NETPATH + '2conv_40X10Filter_256X10_relu.cn')