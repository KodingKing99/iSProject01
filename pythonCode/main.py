from tfl_image_anns import *
print("Hello World")
def trainLayers(myann, 
                data_train_X, 
                data_train_Y,
                data_test_X,
                data_test_Y,
                data_valid_X,
                data_valid_Y,
                dataset):
    train_tfl_image_ann_model(myann,
                            data_train_X, 
                            data_train_Y,
                            data_test_X,
                            data_test_Y,
                            num_epochs=3)
    validD = validate_tfl_image_ann_model(myann,
                                        data_valid_X,
                                        data_valid_Y)
    if(validD > 0.95 or (dataset == "BEE2" and validD > 0.8) or (dataset == "BEE4" and validD > 0.85)):
        # validD = (validD * 100 / 100)
        myann.save(NETPATH + str(dataset) + '256X10_relu_bigboy_v2.3_' + str(validD) +'_valid.tfl')
    print(dataset)
    print(validD)
# def validate3():
def validate3(ann):
    print("validating all three")
    validA = validate_tfl_image_ann_model(ann,
                                        BEE1_gray_valid_X,
                                        BEE1_gray_valid_Y)
    print(f'BEE1: {validA}')
    validB = validate_tfl_image_ann_model(ann,
                                        BEE2_1S_gray_valid_X,
                                        BEE2_1S_gray_valid_Y)
    print(f'BEE2: {validB}')
    validC = validate_tfl_image_ann_model(ann,
                                        BEE4_gray_valid_X,
                                        BEE4_gray_valid_Y)
    print(f'BEE3: {validC}')
    # if validA > 0.7 and validB > 0.7 and validC > 0.7:
    #     # myann.save(NETPATH + '512X256X10_relu_bigboy_75ALL_'+ str(i)+ '.tfl') 
    #     ann.save(NETPATH + '256X10_relu_bigData_70ALL.tfl') 
def makeBigData():
    print("making big data...")
    print("making big_data_train_X...")
    big_data_train_X = BEE1_gray_train_X
    big_data_train_X = np.append(big_data_train_X, BEE2_1S_gray_train_X, axis=0)
    big_data_train_X = np.append(big_data_train_X, BEE4_gray_train_X, axis=0)
    print("DONE!")
    print("making big_data_train_Y...")
    big_data_train_Y = BEE1_gray_train_Y
    big_data_train_Y = np.append(big_data_train_Y, BEE2_1S_gray_train_Y, axis=0)
    big_data_train_Y = np.append(big_data_train_Y, BEE4_gray_train_Y, axis=0)
    print("DONE!")
    print("making big_data_test_X...")
    big_data_test_X = BEE1_gray_test_X
    big_data_test_X = np.append(big_data_test_X, BEE2_1S_gray_test_X, axis=0)
    big_data_test_X = np.append(big_data_test_X, BEE4_gray_test_X, axis=0)
    print("DONE!")
    print("making big_data_test_Y...")
    big_data_test_Y = BEE1_gray_test_Y
    big_data_test_Y = np.append(big_data_test_Y, BEE2_1S_gray_test_Y, axis=0)
    big_data_test_Y = np.append(big_data_test_Y, BEE4_gray_test_Y, axis=0)
    print("DONE!")
    return big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y
    # big_data_train_X = BEE1_gray_train_X
    # big_data_train_X = np.append(big_data_train_X, BEE2_1S_gray_train_X, axis=0)
    # big_data_train_X = np.append(big_data_train_X, BEE4_gray_train_X)
# myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_post_B2.tfl')

big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y = makeBigData()
print(big_data_train_X.shape)
print(big_data_train_Y.shape)
print(big_data_test_X.shape)
print(big_data_test_Y.shape)
# learnrate = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
# for i in learnrate:
#     print(i)
#     myann = make_3_layer_254X10_relu(learn_rate=i)
#     train_tfl_image_ann_model(myann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=3)
#     # del big_data_train_X
#     validate3(myann)
#     myann.save(NETPATH + '256X10_relu_bigData_lr_' + str(i) + '.tfl')
def trainANNs():
    myann = load_and_train_254X10_relu(NETPATH + 'image_ann.tfl', learn_rate=0.04)
    train_tfl_image_ann_model(myann, big_data_train_X, big_data_train_Y, big_data_test_X, big_data_test_Y, num_epochs=20)
    validate3(myann)
    myann.save(NETPATH + 'image_ann.tfl')
def trainConvNets():
    print("training conv nets...")
