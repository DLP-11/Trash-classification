import tensorflow as tf
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

class_names = ['Hazardous waste_Insecticide', 'Hazardous waste_Sphygmometers', 'Hazardous waste_accumulators', 'Hazardous waste_battery', 'Hazardous waste_battery plates', 'Hazardous waste_coin cell batteries', 'Hazardous waste_glue', 'Hazardous waste_lamp', 'Hazardous waste_medicine bottle', 'Hazardous waste_nail polish', 'Hazardous waste_ointment', 'Hazardous waste_pharmaceutical packaging', 'Hazardous waste_pill', 'Hazardous waste_thermometers', 'Kitchen waste_Eight treasure porridge', 'Kitchen waste_Ice candy cane', 'Kitchen waste_apple', 'Kitchen waste_bean', 'Kitchen waste_breads', 'Kitchen waste_cake', 'Kitchen waste_cherry tomato', 'Kitchen waste_chicken wings', 'Kitchen waste_chips', 'Kitchen waste_chocolate', 'Kitchen waste_coffee', 'Kitchen waste_cookies', 'Kitchen waste_dragon fruit', 'Kitchen waste_egg', 'Kitchen waste_egg tart', 'Kitchen waste_fries', 'Kitchen waste_fruit', 'Kitchen waste_garlic', 'Kitchen waste_groundnut', 'Kitchen waste_ham', 'Kitchen waste_ice cream', 'Kitchen waste_jellies', 'Kitchen waste_meats', 'Kitchen waste_melon seeds', 'Kitchen waste_mushroom', 'Kitchen waste_nuts', 'Kitchen waste_orange', 'Kitchen waste_pear', 'Kitchen waste_peel', 'Kitchen waste_peppers', 'Kitchen waste_pickle', 'Kitchen waste_pineapple', 'Kitchen waste_pineapple honey', 'Kitchen waste_radish', 'Kitchen waste_roasted chicken', 'Kitchen waste_sausage', 'Kitchen waste_scraps of leftover food', 'Kitchen waste_shell', 'Kitchen waste_straw bowl', 'Kitchen waste_straw cup', 'Kitchen waste_strawberry', 'Kitchen waste_sugar cane', 'Kitchen waste_tea leaves', 'Kitchen waste_tofu', 'Kitchen waste_tomatoes', 'Kitchen waste_vegetables', 'Kitchen waste_vermicelli', 'Kitchen waste_walnuts', 'Recyclable waste_Air humidifier', 'Recyclable waste_Air purifiers', 'Recyclable waste_Phone', 'Recyclable waste_TVs', 'Recyclable waste_Yoga ball', 'Recyclable waste_accessory', 'Recyclable waste_alarm', 'Recyclable waste_aluminum supplies', 'Recyclable waste_ashtrays', 'Recyclable waste_audio', 'Recyclable waste_bag', 'Recyclable waste_bags', 'Recyclable waste_barrel', 'Recyclable waste_bicycle', 'Recyclable waste_binocular', 'Recyclable waste_blanket', 'Recyclable waste_blow dryer', 'Recyclable waste_boarding pass', 'Recyclable waste_book', 'Recyclable waste_bowls', 'Recyclable waste_box', 'Recyclable waste_boxes', 'Recyclable waste_bracelets', 'Recyclable waste_cage', 'Recyclable waste_calculators', 'Recyclable waste_calendar', 'Recyclable waste_card', 'Recyclable waste_cardboard', 'Recyclable waste_charging cable', 'Recyclable waste_charging head', 'Recyclable waste_charging power', 'Recyclable waste_circuit board', 'Recyclable waste_cling film inner core', 'Recyclable waste_cloth products', 'Recyclable waste_clothes rack', 'Recyclable waste_computer screen', 'Recyclable waste_cushion', 'Recyclable waste_earmuff', 'Recyclable waste_electric fan', 'Recyclable waste_electric hair curling iron', 'Recyclable waste_electric iron', 'Recyclable waste_electric shaver', 'Recyclable waste_electronic scales', 'Recyclable waste_empty bottles of skin care products', 'Recyclable waste_envelope', 'Recyclable waste_filters', 'Recyclable waste_fire extinguishers', 'Recyclable waste_fish tank', 'Recyclable waste_flashlights', 'Recyclable waste_floor sweeper', 'Recyclable waste_foam board', 'Recyclable waste_gas bottle', 'Recyclable waste_gas cooker', 'Recyclable waste_glass ball', 'Recyclable waste_glass pot', 'Recyclable waste_glass products', 'Recyclable waste_glassware', 'Recyclable waste_globe', 'Recyclable waste_hangtags', 'Recyclable waste_hats', 'Recyclable waste_headphones', 'Recyclable waste_hot water bottle', 'Recyclable waste_hula hoop', 'Recyclable waste_induction cooker', 'Recyclable waste_inflator', 'Recyclable waste_jar', 'Recyclable waste_jelly cup', 'Recyclable waste_keyboard', 'Recyclable waste_keys', 'Recyclable waste_knife', 'Recyclable waste_lampshade', 'Recyclable waste_lid', 'Recyclable waste_magnets', 'Recyclable waste_magnifiers', 'Recyclable waste_measuring cup', 'Recyclable waste_metal products', 'Recyclable waste_microphones', 'Recyclable waste_mobile', 'Recyclable waste_mold', 'Recyclable waste_mouse', 'Recyclable waste_nails', 'Recyclable waste_network card', 'Recyclable waste_nylon rope', 'Recyclable waste_pants', 'Recyclable waste_paper products', 'Recyclable waste_pieces', 'Recyclable waste_pillowcase', 'Recyclable waste_ping-pong bat', 'Recyclable waste_placemats', 'Recyclable waste_plastic products', 'Recyclable waste_plates', 'Recyclable waste_pot', 'Recyclable waste_pot lid', 'Recyclable waste_powdered milk bucket', 'Recyclable waste_power strip', 'Recyclable waste_printer', 'Recyclable waste_radio', 'Recyclable waste_rechargeable toothbrush', 'Recyclable waste_remote controls', 'Recyclable waste_rice cooker', 'Recyclable waste_router', 'Recyclable waste_ruler', 'Recyclable waste_scrub board', 'Recyclable waste_shoes', 'Recyclable waste_shot put', 'Recyclable waste_skirt', 'Recyclable waste_slippers', 'Recyclable waste_socks', 'Recyclable waste_sofa', 'Recyclable waste_solar water heater', 'Recyclable waste_soy milk maker', 'Recyclable waste_stainless steel products', 'Recyclable waste_stapling machine', 'Recyclable waste_stool', 'Recyclable waste_subway tickets', 'Recyclable waste_table', 'Recyclable waste_table lamp', 'Recyclable waste_tableware', 'Recyclable waste_thermos', 'Recyclable waste_tires', 'Recyclable waste_toys', 'Recyclable waste_trolley case', 'Recyclable waste_tweezers', 'Recyclable waste_umbrellas', 'Recyclable waste_warm patch', 'Recyclable waste_watches', 'Recyclable waste_water bottle', 'Recyclable waste_water glasses', 'Recyclable waste_weight scale', 'Recyclable waste_wire ball', 'Recyclable waste_wooden carving', 'Recyclable waste_wooden comb', 'Recyclable waste_wooden cutting board', 'Recyclable waste_wooden spatula', 'Recyclable waste_wooden stick', 'Recyclable waste_wrapping rope', 'other waste_Anti-mold and anti-moth tablets', 'other waste_PE plastic bag', 'other waste_U-shaped paper clip', 'other waste_air conditioning filter', 'other waste_alcoholic cotton', 'other waste_band-aid', 'other waste_big lobster head', 'other waste_chicken feather duster', 'other waste_cigarette butt', 'other waste_correction tape', 'other waste_cutting board', 'other waste_dehumidification bag', 'other waste_desiccant', 'other waste_disposable cotton swabs', 'other waste_disposable cups', 'other waste_electric mosquito incense', 'other waste_eyeglass', 'other waste_eyeglass cloth', 'other waste_flyswatter', 'other waste_frothing net', 'other waste_fruit shell', 'other waste_glue waste packaging', 'other waste_kitchen gloves', 'other waste_kitchen wipes', 'other waste_lighter', 'other waste_lottery', 'other waste_lunchbox', 'other waste_mask', 'other waste_milk tea cup', 'other waste_movie tickets', 'other waste_paper napkin', 'other waste_pasteurized cloth', 'other waste_pen', 'other waste_pregnancy test', 'other waste_recordings', 'other waste_scrubbing towel', 'other waste_skewer bamboo skewer', 'other waste_sticky note', 'other waste_straw hat', 'other waste_tapes', 'other waste_teapot fragments', 'other waste_thumbtack', 'other waste_tickets', 'other waste_toilet paper', 'other waste_toothbrushes', 'other waste_towel', 'other waste_wet paper towel']

cnn_modle_path = 'models/cnn_245_epoch30.h5'
mobilenet_path = 'models/mobilenet_245_epoch30.h5'
data_path = 'trash_jpg_rename'



# 数据加载
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def test_mobilenet():
    train_ds, val_ds, class_names = data_load(data_path, 224, 224, 128)
    model = tf.keras.models.load_model(mobilenet_path)
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    with open("results/mobilenet_test.txt", 'w') as f:
      print('Mobilenet test accuracy :', accuracy, file=f)
    zhibiao(data_path, mobilenet_path)



def test_cnn():
    train_ds, val_ds, class_names = data_load(data_path, 224, 224, 128)
    model = tf.keras.models.load_model(cnn_modle_path)
    model.summary()
    loss, accuracy = model.evaluate(val_ds)
    with open("results/cnn_test.txt", 'w') as f:
      print('CNN test accuracy :', accuracy, file=f)
    zhibiao(data_path, cnn_modle_path)


def zhibiao(folder_name, model_path):
    # 遍历文件夹返回数目
    trash_names = ['Kitchen waste', 'Recyclable waste', 'other waste', 'Hazardous waste']
    real_label = []
    pre_label = []
    images_path = []
    folders = os.listdir(folder_name)

    for folder in folders:
        folder_path = os.path.join(folder_name, folder)
        images = os.listdir(folder_path)
        for img in images:
            xxx = folder.split("_")[0]
            x_idx = trash_names.index(xxx)
            img_path = os.path.join(folder_path, img)
            real_label.append(x_idx)
            images_path.append(img_path)

    model = tf.keras.models.load_model(model_path)
    for ii, i_path in enumerate(images_path):
        print("{}/{}".format(ii, len(images_path) - 1))
        shutil.copy(i_path, "images/t1.jpg")
        src_i = cv2.imread("images/t1.jpg")
        src_r = cv2.resize(src_i, (224, 224))
        cv2.imwrite("images/t2.jpg", src_r)
        img = Image.open("images/t2.jpg")
        img = np.asarray(img)
        outputs = model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = class_names[result_index]
        names = result.split("_")
        xxx = names[0]
        x_idx = trash_names.index(xxx)
        pre_label.append(x_idx)

    with open("results/{}_test.txt".format(model_path), 'w') as f:
        print('done', file=f)
        print(pre_label, file=f)
        print(real_label, file=f)
        print(classification_report(real_label, pre_label, target_names=trash_names), file=f)



if __name__ == '__main__':
    test_cnn()
    test_mobilenet()

