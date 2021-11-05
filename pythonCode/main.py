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

def makeBigData():
    big_data_train_X = BEE1_gray_train_X
    big_data_train_X = np.append(big_data_train_X, BEE2_1S_gray_train_X, axis=0)
    big_data_train_X = np.append(big_data_train_X, BEE4_gray_train_X, axis=0)
    big_data_train_Y = BEE1_gray_train_Y
    big_data_train_Y = np.append(big_data_train_Y, BEE2_1S_gray_train_Y, axis=0)
    big_data_train_Y = np.append(big_data_train_Y, BEE4_gray_train_Y, axis=0)
    return big_data_train_X, big_data_train_Y
    # big_data_train_X = BEE1_gray_train_X
    # big_data_train_X = np.append(big_data_train_X, BEE2_1S_gray_train_X, axis=0)
    # big_data_train_X = np.append(big_data_train_X, BEE4_gray_train_X)
# myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_post_B2.tfl')
myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_bigboy_v2.2.tfl', learn_rate=0.03)
# myann = make_4_layer_512X256X10X2_relu(learn_rate=0.035)
for i in range(15):
    trainLayers(myann, 
                BEE2_1S_gray_train_X, 
                BEE2_1S_gray_train_Y, 
                BEE2_1S_gray_test_X, 
                BEE2_1S_gray_test_Y,
                BEE2_1S_gray_valid_X,
                BEE2_1S_gray_valid_Y,
                "BEE2")
    trainLayers(myann, 
                BEE1_gray_train_X, 
                BEE1_gray_train_Y, 
                BEE1_gray_test_X, 
                BEE1_gray_test_Y,
                BEE1_gray_valid_X,
                BEE1_gray_valid_Y,
                "BEE1")
    trainLayers(myann, 
                BEE4_gray_train_X, 
                BEE4_gray_train_Y, 
                BEE4_gray_test_X, 
                BEE4_gray_test_Y,
                BEE4_gray_valid_X,
                BEE4_gray_valid_Y,
                "BEE4")
    print("validating all three")
    validA = validate_tfl_image_ann_model(myann,
                                        BEE1_gray_valid_X,
                                        BEE1_gray_valid_Y)
    print(f'BEE1: {validA}')
    validB = validate_tfl_image_ann_model(myann,
                                        BEE2_1S_gray_valid_X,
                                        BEE2_1S_gray_valid_Y)
    print(f'BEE2: {validB}')
    validC = validate_tfl_image_ann_model(myann,
                                        BEE4_gray_valid_X,
                                        BEE4_gray_valid_Y)
    print(f'BEE3: {validC}')
    if validA > 0.7 and validB > 0.7 and validC > 0.7:
        # myann.save(NETPATH + '512X256X10_relu_bigboy_75ALL_'+ str(i)+ '.tfl') 
        myann.save(NETPATH + '256X10_relu_bigboy_70ALL_'+ str(i)+ '.tfl') 
# myann.save(NETPATH + '512X256X10_relu_bigboy_v1.tfl')
myann.save(NETPATH + '256X10_relu_bigboy_v2.3.tfl')