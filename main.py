import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from io import BytesIO
import json
import base64
app = Flask(__name__)
#CORS(app)  # 解决跨域问题

names = ['Hazardous waste_accumulator', 'Hazardous waste_button battery', 'Hazardous waste_cell', 'Hazardous waste_cell panel', 'Hazardous waste_drug packaging',
         'Hazardous waste_light', 'Hazardous waste_medicine bottle', 'Hazardous waste_mucilage', 'Hazardous waste_nail enamel', 'Hazardous waste_pesticide',
         'Hazardous waste_sphygmomanometer', 'Hazardous waste_tablet', 'Hazardous waste_thermometer', 'Hazardous waste_unguent', 'Kitchen waste _ apple',
         'Kitchen waste _ Babao porridge', 'Kitchen waste _ beans', 'Kitchen waste _ biscuits', 'Kitchen waste _ bread', 'Kitchen waste _ cake', 'Kitchen waste _ chicken wings',
         'Kitchen waste _ chips', 'Kitchen waste _ chocolate', 'Kitchen waste _ coffee', 'Kitchen waste _ dragon fruit', 'Kitchen waste _ egg', 'Kitchen waste _ egg tart',
         'Kitchen waste _ fruit', 'Kitchen waste _ garlic', 'Kitchen waste _ ham', 'Kitchen waste _ holy fruit', 'Kitchen waste _ ice cream', 'Kitchen waste _ jelly',
         'Kitchen waste _ meat', 'Kitchen waste _ mushroom', 'Kitchen waste _ nut', 'Kitchen waste _ orange', 'Kitchen waste _ pear', 'Kitchen waste _ peel',
         'Kitchen waste _ pepper', 'Kitchen waste _ pickle', 'Kitchen waste _ pineapple', 'Kitchen waste _ pineapple honey', 'Kitchen waste _ potato chips',
         'Kitchen waste _ radish', 'Kitchen waste _ residue leftover', 'Kitchen waste _ roast chicken', 'Kitchen waste _ sausage', 'Kitchen waste _ shell',
         'Kitchen waste _ straw bowl', 'Kitchen waste _ straw cup', 'Kitchen waste _ strawberry', 'Kitchen waste _ sugar gourd', 'Kitchen waste _ sugarcane',
         'Kitchen waste _ sunflower seeds', 'Kitchen waste _ sweet potato', 'Kitchen waste _ tea', 'Kitchen waste _ tofu', 'Kitchen waste _ tomato', 'Kitchen waste _ vegetables',
         'Kitchen waste _ vermicelli', 'Kitchen waste _ walnut', 'Other waste_adhesive tape', 'Other waste_air filter', 'Other waste_Alcohol cotton','Other waste_baid-aid',
         'Other waste_bath towel', 'Other waste_Bubbling network', 'Other waste_chicken feather duster', 'Other waste_Clean cloth', 'Other waste_Coating belt',
         'Other waste_convenient paste', 'Other waste_cutting board', 'Other waste_Dehumidifying bag', 'Other waste_desiccant', 'Other waste_drawing pin',
         'Other waste_Dry fruit shell', 'Other waste_Eyeglass cloth', 'Other waste_film ticket', 'Other waste_fly swatter', 'Other waste_Kitchen cloth', 'Other waste_Kitchen gloves',
         'Other waste_lighter', 'Other waste_lottery ticket', 'Other waste_lunch box', 'Other waste_mask', 'Other waste_mildewproof sheet', 'Other waste_milky tea cups',
         'Other waste_mosquito incense', 'Other waste_napkin paper', 'Other waste_One-time cotton swab', 'Other waste_One-time cup', 'Other waste_PE plastic bag',
         'Other waste_pen', 'Other waste_pregnancy test stick', 'Other waste_prod', 'Other waste_record', 'Other waste_roach', 'Other waste_Rubber waste packaging',
         'Other waste_Shrimp head', 'Other waste_spectacles', 'Other waste_straw hat', 'Other waste_Teapot fragments', 'Other waste_ticket', 'Other waste_toilet paper',
         'Other waste_toothbrush', 'Other waste_towel', 'Other waste_U-shaped needle', 'Other waste_wet tissues', 'Recyclable waste_air humidifier', 'Recyclable waste_air purifier',
         'Recyclable waste_alarm bell', 'Recyclable waste_Aluminum supplies', 'Recyclable waste_ashtray', 'Recyclable waste_bag', 'Recyclable waste_Barrels',
         'Recyclable waste_bench', 'Recyclable waste_blanket', 'Recyclable waste_blower', 'Recyclable waste_boarding check', 'Recyclable waste_book', 'Recyclable waste_bowl',
         'Recyclable waste_Box', 'Recyclable waste_bracelet', 'Recyclable waste_cage', 'Recyclable waste_calculator', 'Recyclable waste_Calendar', 'Recyclable waste_calorie',
         'Recyclable waste_Cards', 'Recyclable waste_cassette', 'Recyclable waste_charge line', 'Recyclable waste_Charging head', 'Recyclable waste_charging po',
         'Recyclable waste_Charging toothbrush', 'Recyclable waste_Chess', 'Recyclable waste_circuit board', 'Recyclable waste_computer screen', 'Recyclable waste_cushion',
         'Recyclable waste_draw-bar case', 'Recyclable waste_earcap', 'Recyclable waste_earphone', 'Recyclable waste_Electric curling rod', 'Recyclable waste_electric iron',
         'Recyclable waste_electric razor', 'Recyclable waste_electromagnetic oven', 'Recyclable waste_electronic scale', 'Recyclable waste_Envelope', 'Recyclable waste_Fabrics',
         'Recyclable waste_fanner', 'Recyclable waste_Fire extinguisher', 'Recyclable waste_Fish tank', 'Recyclable waste_flashlight', 'Recyclable waste_Foamboard',
         'Recyclable waste_Frozen cup', 'Recyclable waste_Gas bottles', 'Recyclable waste_Gas stove', 'Recyclable waste_glass ball', 'Recyclable waste_Glass pot',
         'Recyclable waste_glass ware', 'Recyclable waste_glassware', 'Recyclable waste_globe', 'Recyclable waste_hand tag', 'Recyclable waste_handbag', 'Recyclable waste_hanger',
         'Recyclable waste_Hat', 'Recyclable waste_Heat paste', 'Recyclable waste_Hot kettle', 'Recyclable waste_Hot water bottle', 'Recyclable waste_hula hoop',
         'Recyclable waste_Inner core of preservative film', 'Recyclable waste_Iron ball', 'Recyclable waste_jar', 'Recyclable waste_kettle', 'Recyclable waste_Key',
         'Recyclable waste_keyboard', 'Recyclable waste_knife', 'Recyclable waste_lampshade', 'Recyclable waste_Lead ball', 'Recyclable waste_lid', 'Recyclable waste_magnet',
         'Recyclable waste_magnifier', 'Recyclable waste_manual tyre pump', 'Recyclable waste_measuring cup', 'Recyclable waste_metalwork', 'Recyclable waste_microphone',
         'Recyclable waste_Milk powder bucket', 'Recyclable waste_Mobile phone', 'Recyclable waste_Mould', 'Recyclable waste_Mouse', 'Recyclable waste_nail',
         'Recyclable waste_Network card', 'Recyclable waste_Nylon rope', 'Recyclable waste_Ornaments', 'Recyclable waste_pant', 'Recyclable waste_Paper products',
         'Recyclable waste_Pillow', 'Recyclable waste_Plastic products', 'Recyclable waste_Plate', 'Recyclable waste_printer', 'Recyclable waste_Radio',
         'Recyclable waste_Remote controller', 'Recyclable waste_rice cooker', 'Recyclable waste_rope', 'Recyclable waste_Router', 'Recyclable waste_ruler',
         'Recyclable waste_Shoes', 'Recyclable waste_single chassis unit', 'Recyclable waste_Skincare empty bottles', 'Recyclable waste_Skirt', 'Recyclable waste_Slippers',
         'Recyclable waste_Socks', 'Recyclable waste_Sofa', 'Recyclable waste_Solar water heater', 'Recyclable waste_Sound', 'Recyclable waste_soymilk maker',
         'Recyclable waste_stainless steel product', 'Recyclable waste_stapler', 'Recyclable waste_sticks', 'Recyclable waste_strainer', 'Recyclable waste_Subway tickets',
         'Recyclable waste_Sweeping robot', 'Recyclable waste_Table', 'Recyclable waste_Table lamp', 'Recyclable waste_table mat', 'Recyclable waste_Table tennis racket',
         'Recyclable waste_tableware', 'Recyclable waste_telephone', 'Recyclable waste_Telescope', 'Recyclable waste_television set', 'Recyclable waste_Tires',
         'Recyclable waste_Toys', 'Recyclable waste_Tweezers', 'Recyclable waste_Umbrella', 'Recyclable waste_vacuum flask', 'Recyclable waste_washboard',
         'Recyclable waste_Watches', 'Recyclable waste_Water cup', 'Recyclable waste_Weight scale', 'Recyclable waste_wiring board', 'Recyclable waste_wok lid',
         'Recyclable waste_Wood carving', 'Recyclable waste_Wood-cut vegetable plates', 'Recyclable waste_Wooden comb', 'Recyclable waste_Wooden spade', 'Recyclable waste_Yoga ball']



def single_img_predict(img, model_path='./models/my_train_mobilenet_trashv1_2_50.pth'):#img_path="imgs/6.png"
    # 加载网络
    print('------------------------------------------')
    print(type(img))
    print('-------------------------------------------')
    net = torch.load(model_path, map_location=lambda storage, loc: storage)
    transform = transforms.Compose(
        # 这里只对其中的一个通道进行归一化的操作
        # [transforms.RandomResizedCrop(224),
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img_torch = transform(img)
    img_torch = img_torch.view(-1, 3, 224, 224)

    # 开始预测
    outputs = net(img_torch)
    _, predicted = torch.max(outputs, 1)
    temp_info = [names[predicted[0].numpy()]]
    return_info = {"result": temp_info}
    print(return_info)
    return return_info


@app.route("/predict", methods=["POST"])#
def predict():
    image = base64.b64decode(request.form.get("image"))
    bytes_stream = BytesIO(image)
    image = Image.open(bytes_stream)
    RGB_img = image.convert('RGB')
    info = single_img_predict(img = RGB_img)
    return jsonify(info)


@app.route("/")#, methods=["GET", "POST"]
def root():
    return 'Hellow Word'  


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)  # 255.255.255.240   0.0.0.0