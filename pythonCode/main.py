from tfl_image_anns import *
print("Hello World")
# myann = load_and_train_254X10_relu(NETPATH + '256X10_relu_post_B2.tfl')
myann = make_3_layer_254X10_relu(0.04)
for i in range(3):
    train_tfl_image_ann_model(myann,
                             BEE1_gray_train_X, 
                             BEE1_gray_train_Y, 
                             BEE1_gray_test_X, 
                             BEE1_gray_test_Y,
                             num_epochs=5)
    print(validate_tfl_image_ann_model(myann,
                                    BEE1_gray_valid_X,
                                    BEE1_gray_valid_Y))
    train_tfl_image_ann_model(myann,
                             BEE2_1S_gray_train_X, 
                             BEE2_1S_gray_train_Y, 
                             BEE2_1S_gray_test_X, 
                             BEE2_1S_gray_test_Y,
                             num_epochs=5) 
    print(validate_tfl_image_ann_model(myann,
                                    BEE2_1S_gray_valid_X,
                                    BEE2_1S_gray_valid_Y))
    train_tfl_image_ann_model(myann,
                             BEE4_gray_train_X, 
                             BEE4_gray_train_Y, 
                             BEE4_gray_test_X, 
                             BEE4_gray_test_Y,
                             num_epochs=5)
    print(validate_tfl_image_ann_model(myann,
                                    BEE4_gray_valid_X,
                                    BEE4_gray_valid_Y))
myann.save(NETPATH + '256X10_relu_bigboy.tfl')