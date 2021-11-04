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
                            num_epochs=5)
    validD = validate_tfl_image_ann_model(myann,
                                        data_valid_X,
                                        data_valid_Y)
    if(validD > 0.9):
        myann.save(NETPATH + dataset + '256X10_relu_bigboy_v2_' + (validD * 100 / 100) +'_valid.tfl')
    print(dataset)
    print(validD)


# myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_post_B2.tfl')
myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_bigboy.tfl', learn_rate=0.037)
for i in range(5):
    trainLayers(myann, 
                BEE1_gray_train_X, 
                BEE1_gray_train_Y, 
                BEE1_gray_test_X, 
                BEE1_gray_test_Y,
                BEE1_gray_valid_X,
                BEE1_gray_valid_Y,
                "BEE1")
    trainLayers(myann, 
                BEE2_1S_gray_train_X, 
                BEE2_1S_gray_train_Y, 
                BEE2_1S_gray_test_X, 
                BEE2_1S_gray_test_Y,
                BEE2_1S_gray_valid_X,
                BEE2_1S_gray_valid_Y,
                "BEE2")
    trainLayers(myann, 
                BEE4_gray_train_X, 
                BEE4_gray_train_Y, 
                BEE4_gray_test_X, 
                BEE4_gray_test_Y,
                BEE4_gray_valid_X,
                BEE4_gray_valid_Y,
                "BEE4")
myann.save(NETPATH + '256X10_relu_bigboy_v2.tfl')