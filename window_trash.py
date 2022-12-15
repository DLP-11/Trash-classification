import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil



class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo1.png'))
        self.setWindowTitle('Intelligent Waste Classification System')
        self.model = tf.keras.models.load_model("models/mobilenet_245_epoch30.h5")
        self.to_predict_name = "images/add.png"
        self.class_names = ['Hazardous waste_Insecticide', 'Hazardous waste_Sphygmometers', 'Hazardous waste_accumulators', 'Hazardous waste_battery', 'Hazardous waste_battery plates',
               'Hazardous waste_coin cell batteries', 'Hazardous waste_glue', 'Hazardous waste_lamp', 'Hazardous waste_medicine bottle', 'Hazardous waste_nail polish',
               'Hazardous waste_ointment', 'Hazardous waste_pharmaceutical packaging', 'Hazardous waste_pill', 'Hazardous waste_thermometers', 'Kitchen waste_Eight treasure porridge',
               'Kitchen waste_Ice candy cane', 'Kitchen waste_apple', 'Kitchen waste_bean', 'Kitchen waste_breads', 'Kitchen waste_cake', 'Kitchen waste_cherry tomato',
               'Kitchen waste_chicken wings', 'Kitchen waste_chips', 'Kitchen waste_chocolate', 'Kitchen waste_coffee', 'Kitchen waste_cookies', 'Kitchen waste_dragon fruit',
               'Kitchen waste_egg', 'Kitchen waste_egg tart', 'Kitchen waste_fries', 'Kitchen waste_fruit', 'Kitchen waste_garlic', 'Kitchen waste_groundnut', 'Kitchen waste_ham',
               'Kitchen waste_ice cream', 'Kitchen waste_jellies', 'Kitchen waste_meats', 'Kitchen waste_melon seeds', 'Kitchen waste_mushroom', 'Kitchen waste_nuts',
               'Kitchen waste_orange', 'Kitchen waste_pear', 'Kitchen waste_peel', 'Kitchen waste_peppers', 'Kitchen waste_pickle', 'Kitchen waste_pineapple',
               'Kitchen waste_pineapple honey', 'Kitchen waste_radish', 'Kitchen waste_roasted chicken', 'Kitchen waste_sausage', 'Kitchen waste_scraps of leftover food',
               'Kitchen waste_shell', 'Kitchen waste_straw bowl', 'Kitchen waste_straw cup', 'Kitchen waste_strawberry', 'Kitchen waste_sugar cane', 'Kitchen waste_tea leaves',
               'Kitchen waste_tofu', 'Kitchen waste_tomatoes', 'Kitchen waste_vegetables', 'Kitchen waste_vermicelli', 'Kitchen waste_walnuts', 'Recyclable waste_Air humidifier',
               'Recyclable waste_Air purifiers', 'Recyclable waste_Phone', 'Recyclable waste_TVs', 'Recyclable waste_Yoga ball', 'Recyclable waste_accessory',
               'Recyclable waste_alarm', 'Recyclable waste_aluminum supplies', 'Recyclable waste_ashtrays', 'Recyclable waste_audio', 'Recyclable waste_bag',
               'Recyclable waste_bags', 'Recyclable waste_barrel', 'Recyclable waste_bicycle', 'Recyclable waste_binocular', 'Recyclable waste_blanket',
               'Recyclable waste_blow dryer', 'Recyclable waste_boarding pass', 'Recyclable waste_book', 'Recyclable waste_bowls', 'Recyclable waste_box', 'Recyclable waste_boxes',
               'Recyclable waste_bracelets', 'Recyclable waste_cage', 'Recyclable waste_calculators', 'Recyclable waste_calendar', 'Recyclable waste_card',
               'Recyclable waste_cardboard', 'Recyclable waste_charging cable', 'Recyclable waste_charging head', 'Recyclable waste_charging power', 'Recyclable waste_circuit board',
               'Recyclable waste_cling film inner core', 'Recyclable waste_cloth products', 'Recyclable waste_clothes rack', 'Recyclable waste_computer screen',
               'Recyclable waste_cushion', 'Recyclable waste_earmuff', 'Recyclable waste_electric fan', 'Recyclable waste_electric hair curling iron', 'Recyclable waste_electric iron',
               'Recyclable waste_electric shaver', 'Recyclable waste_electronic scales', 'Recyclable waste_empty bottles of skin care products', 'Recyclable waste_envelope',
               'Recyclable waste_filters', 'Recyclable waste_fire extinguishers', 'Recyclable waste_fish tank', 'Recyclable waste_flashlights', 'Recyclable waste_floor sweeper',
               'Recyclable waste_foam board', 'Recyclable waste_gas bottle', 'Recyclable waste_gas cooker', 'Recyclable waste_glass ball', 'Recyclable waste_glass pot',
               'Recyclable waste_glass products', 'Recyclable waste_glassware', 'Recyclable waste_globe', 'Recyclable waste_hangtags', 'Recyclable waste_hats',
               'Recyclable waste_headphones', 'Recyclable waste_hot water bottle', 'Recyclable waste_hula hoop', 'Recyclable waste_induction cooker', 'Recyclable waste_inflator',
               'Recyclable waste_jar', 'Recyclable waste_jelly cup', 'Recyclable waste_keyboard', 'Recyclable waste_keys', 'Recyclable waste_knife', 'Recyclable waste_lampshade',
               'Recyclable waste_lid', 'Recyclable waste_magnets', 'Recyclable waste_magnifiers', 'Recyclable waste_measuring cup', 'Recyclable waste_metal products',
               'Recyclable waste_microphones', 'Recyclable waste_mobile', 'Recyclable waste_mold', 'Recyclable waste_mouse', 'Recyclable waste_nails', 'Recyclable waste_network card',
               'Recyclable waste_nylon rope', 'Recyclable waste_pants', 'Recyclable waste_paper products', 'Recyclable waste_pieces', 'Recyclable waste_pillowcase',
               'Recyclable waste_ping-pong bat', 'Recyclable waste_placemats', 'Recyclable waste_plastic products', 'Recyclable waste_plates', 'Recyclable waste_pot',
               'Recyclable waste_pot lid', 'Recyclable waste_powdered milk bucket', 'Recyclable waste_power strip', 'Recyclable waste_printer', 'Recyclable waste_radio',
               'Recyclable waste_rechargeable toothbrush', 'Recyclable waste_remote controls', 'Recyclable waste_rice cooker', 'Recyclable waste_router', 'Recyclable waste_ruler',
               'Recyclable waste_scrub board', 'Recyclable waste_shoes', 'Recyclable waste_shot put', 'Recyclable waste_skirt', 'Recyclable waste_slippers', 'Recyclable waste_socks',
               'Recyclable waste_sofa', 'Recyclable waste_solar water heater', 'Recyclable waste_soy milk maker', 'Recyclable waste_stainless steel products',
               'Recyclable waste_stapling machine', 'Recyclable waste_stool', 'Recyclable waste_subway tickets', 'Recyclable waste_table', 'Recyclable waste_table lamp',
               'Recyclable waste_tableware', 'Recyclable waste_thermos', 'Recyclable waste_tires', 'Recyclable waste_toys', 'Recyclable waste_trolley case', 'Recyclable waste_tweezers',
               'Recyclable waste_umbrellas', 'Recyclable waste_warm patch', 'Recyclable waste_watches', 'Recyclable waste_water bottle', 'Recyclable waste_water glasses',
               'Recyclable waste_weight scale', 'Recyclable waste_wire ball', 'Recyclable waste_wooden carving', 'Recyclable waste_wooden comb', 'Recyclable waste_wooden cutting board',
               'Recyclable waste_wooden spatula', 'Recyclable waste_wooden stick', 'Recyclable waste_wrapping rope', 'other waste_Anti-mold and anti-moth tablets',
               'other waste_PE plastic bag', 'other waste_U-shaped paper clip', 'other waste_air conditioning filter', 'other waste_alcoholic cotton', 'other waste_band-aid',
               'other waste_big lobster head', 'other waste_chicken feather duster', 'other waste_cigarette butt', 'other waste_correction tape', 'other waste_cutting board',
               'other waste_dehumidification bag', 'other waste_desiccant', 'other waste_disposable cotton swabs', 'other waste_disposable cups',
               'other waste_electric mosquito incense', 'other waste_eyeglass', 'other waste_eyeglass cloth', 'other waste_flyswatter', 'other waste_frothing net',
               'other waste_fruit shell', 'other waste_glue waste packaging', 'other waste_kitchen gloves', 'other waste_kitchen wipes', 'other waste_lighter', 'other waste_lottery',
               'other waste_lunchbox', 'other waste_mask', 'other waste_milk tea cup', 'other waste_movie tickets', 'other waste_paper napkin', 'other waste_pasteurized cloth',
               'other waste_pen', 'other waste_pregnancy test', 'other waste_recordings', 'other waste_scrubbing towel', 'other waste_skewer bamboo skewer', 'other waste_sticky note',
               'other waste_straw hat', 'other waste_tapes', 'other waste_teapot fragments', 'other waste_thumbtack', 'other waste_tickets', 'other waste_toilet paper',
               'other waste_toothbrushes', 'other waste_towel', 'other waste_wet paper towel']
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("Sample")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" Upload images ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" Start classification ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel(' Type of this waste ')
        self.result = QLabel("Waiting for classification")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))

        label_result_f = QLabel(' Item Name ')
        self.result_f = QLabel("Waiting classification")

        self.label_info = QTextEdit()
        self.label_info.setFont(QFont('楷体', 12))

        label_result_f.setFont(QFont('楷体', 16))
        self.result_f.setFont(QFont('楷体', 24))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.label_info, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('Welcome to the waste classification system')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/about.png'))
        about_img.setAlignment(Qt.AlignCenter)
        #label_super = QLabel('<a href="https://space.bilibili.com/161240964">作者：dejahu（关注我不迷路）</a>')
        #label_super.setFont(QFont('楷体', 12))
        #label_super.setOpenExternalLinks(True)
        #label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        #about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, 'Home')
        # self.addTab(about_widget, 'regarding')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmpx.jpg"
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))

    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        names = result.split("_")
        # print(result)
        if names[0] == "Kitchen waste":
            self.label_info.setText("Kitchen waste is the garbage generated from residents' daily life and activities such as food processing, catering service and unit feeding, including discarded leaves, leftovers, leftovers, peels, eggshells, tea dregs, bones, etc. Since food waste contains extremely high moisture and organic matter, it can easily decay and produce bad odor. After proper treatment and processing, it can be transformed into a new resource. The high organic matter content makes it possible to be used as fertilizer and feed after strict treatment, or to produce biogas for fuel or power generation, and the grease fraction can be used to prepare biofuel.")
        if names[0] == "Hazardous waste":
            self.label_info.setText("Hazardous waste refers to household waste that causes direct or potential harm to human health or the natural environment. Common hazardous waste includes waste lamps, waste paint, pesticides, waste cosmetics, expired drugs, waste batteries, waste light bulbs, waste water silver thermometers, etc. Hazardous waste needs to be disposed of safely in accordance with special and proper methods, and generally requires special treatment before it can be incinerated, composted, or landfilled.")
        if names[0] == "Recyclable waste":
            self.label_info.setText(' According to the industry standard of "Classification of Municipal Domestic Waste and its Evaluation Standard" and the reference to German waste classification method, recyclable waste refers to waste that is suitable for recycling and resource utilization. Mainly includes: paper, plastic, metal, glass, fabric, etc. The main treatment methods are: 1. waste recycling method; 2. waste incineration method; 3. waste composting method; 4. waste biodegradation method.')
        if names[0] == "other waste":
            self.label_info.setText("Other waste refers to waste that is less harmful and has no value for reuse. Other waste includes brick and ceramic, slag, toilet waste paper, porcelain fragments, animal excrement, disposables and other waste that is difficult to recycle. Generally, they are treated by landfill, incineration, sanitary decomposition, etc. Some of them can also be solved by using biodegradation methods.")
        self.result.setText(names[0])
        self.result_f.setText(names[1])

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'Exit',
                                     "Should you exit the program?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
