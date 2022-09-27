// pages/news/news.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    imglist : [],
    request_data:[]
    // date:[],
  },
  
  

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  },
  // a:wx.showActionSheet({
  //   itemList: ['从手机相册选择', '拍照'],
  //   success: function(res) {
  //     console.log(res.tapIndex)
  //   },
  //   fail: function(res) {
  //     console.log(res.errMsg)
  //   }
  // }),
  img_w_show(){
    var _this=this;
    wx.chooseImage({
      count: 9, // 默认9
      sizeType: ['original', 'compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success: function (res) {
        // 返回选定照片的本地文件路径列表，tempFilePath可以作为img标签的src属性显示图片
        var tempFilePaths = res.tempFilePaths;
        var team_image = wx.getFileSystemManager().readFileSync(tempFilePaths[0], "base64") //将图片进行base64编码。
        wx.request({
　　　　　　　　　url: 'http://192.168.10.108:5000/predict',//API地址
　　　　　　　　  header: {
　　　　　　　　　'content-type': "application/x-www-form-urlencoded",
　　　　　　　　  },
　　　　　　　　  data: {
　　　　　　　　　image: team_image,
　　　　　　　　},
               method: "POST",
　　　　　　　　success: function (reg) {
                _this.setData({
                imglist: tempFilePaths//_this.data.imglist.concat(
                })
               _this.setData({
                request_data: reg.data.result
                })
　　　　　　　　　　console.log(reg.data.result)
　　　　　　　　}
　　　　　　　　})
        // wx.uploadFile({
        //    url:"http://10.51.55.125/predict",//weixin
        //   filePath: tempFilePaths[0],
        //   name: 'file',
        //   header:{"Content-Type":"multipart/from-data"},
        //   formData: {
        //     "usr":'donald_trump.jpg'
        //   },
        //   success: function (res) {
        //     _this.setData({
        //     imglist: _this.data.imglist.concat(tempFilePaths)
        //     })
            
        //   },
        //   fail: function (res) {
        //     wx.hideToast();
        //     wx.showModal({
        //       title: '错误提示',
        //       content: '上传图片失败',
        //       showCancel: false,
        //       success: function (res) { }
        //     })
        //   }
        // })
        // wx.request({
        //   url: 'http://localhost:5000/predict', //调用后台接口的全路径/weixin    http://192.168.10.103:5000/predict
        //   data: {//发送给后台的数据
        //   },
        //   // dataType: 'json',
        //   header: {
        //     // 'Content-type': '',
        //     "Content-Type": "application/json"//application/x-www-form-urlencoded
        //   },
        //   method: "post",
        //   success: function (res) { 
        //     console.log(res.data);//res.data相当于ajax里面的data,为后台返回的数据
        //     // that.setData({//如果在sucess直接写this就变成了wx.request()的this了.必须为getdata函数的this,不然无法重置调用函数
        //     //   date: res.data.data  //
        //     // })
        //   },
        //   fail: function (err) { console.log(err.data); },//请求失败
    
        //   complete: function () { }//请求完成后执行的函数
        // })
      }
    })
  }
  


})