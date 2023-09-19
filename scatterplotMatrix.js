/**
 *
 * 将构建散点图矩阵的核心代码封装为一个函数
 *
 */

 let selectedData = [];
 let brush_ids = new Array();
 const buttonValue = document.getElementById("select2");
 var path = 'http://127.0.0.1:7000';
 var url = path+'/api/data/TSP';
 query_stb={};
 query_bts={};
 let brush_ticks = [];
 query_position={};
 
 const highlightedIndices = [];  // 要高亮的点的索引


 function send_brush_data(data, f_svg){
     
     brush_ticks.splice(0,brush_ticks.length)
     query_position={};
     query_bts={};
     query_stb={};
     const pointsGroup = d3.select("#points");
     const pointsGroup1 = d3.select("#pointsa");
     const barChartGroup = d3.select(".brush");
     pointsGroup.selectAll("circle").remove(); // Clear existing circles
     pointsGroup.selectAll("image").remove(); // Clear existing images
     pointsGroup.selectAll("rect").remove();
     pointsGroup1.selectAll("circle").remove();
     barChartGroup.remove();
     d3.select('#line_arrow_svg').selectAll(".bezier").remove();   // 每次可以重新画
    //  d3.select('#barChartGroup').select(".brush").call(brush.move, null);

     //canvas初始化和清洗
     const canvas = document.getElementById('imageCanvas');
     const ctx = canvas.getContext('2d');
     var imageSize = 64;
     const imagesPerRow = 19;
     const totalImages = data.length;
     const numRows = Math.ceil(totalImages / imagesPerRow);
     const spacing = 15.5; // 间隙的大小
     const borderColor = 'red'; // 边框的颜色
     const borderWidth = 4; // 边框的宽度
     canvas.height = numRows * (imageSize + spacing) + spacing; // 调整画布高度
     canvas.width = 1530;
     ctx.clearRect(0, 0, canvas.width, canvas.height);
 
     
     let brush_data = [];
     for (let i = 0; i < brush_ids.length; i++) {
         for (let j = 0; j < data.length; j++) {
           if (brush_ids[i] === data[j].idx) {
             brush_data.push([data[j].idx, data[j].x, data[j].y]);
           }
         }
     };
 
     console.log("brush_data: ", brush_data);
     console.log(url);



     

     // 配置请求参数：设置请求的URL、HTTP方法（GET、POST、PUT等）、请求头和请求体等参数，以便将数据正确发送到后端。
     // 发送请求：通过调用请求对象的send()方法将请求发送到后端
     fetch(url, {
         method: "POST",
         headers: {
           "Content-Type": "application/json"
         },
         body: JSON.stringify(brush_data)
     })
     .then(response => {
         return response.json()
     })
     .then(data => {
         // 处理响应数据
         pointsGroup.innerHTML = "";
         pointsGroup1.innerHTML = "";


         //画总的选中图片***************************

        async function loadImage(src) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = reject;
                img.src = src;
            });
        }

        async function drawImages(data) {
            let currentImageIndex = 0;
            for (let row = 0; row < numRows; row++) {
                for (let col = 0; col < imagesPerRow; col++) {
                    if (currentImageIndex >= totalImages) {
                        return;
                    }

                    const x = col * (imageSize + spacing) + spacing;
                    const y = row * (imageSize + spacing) + spacing;

                    const imgSrc = `cifar10/images256/cf10_image_${data[currentImageIndex][0]}.jpg`;
                    const img = await loadImage(imgSrc);

              // 绘制边框
              ctx.strokeStyle = data[currentImageIndex][3];
              ctx.lineWidth = borderWidth;
              ctx.strokeRect(x, y, imageSize, imageSize);

              // 绘制图片
              ctx.drawImage(img, x + borderWidth, y + borderWidth, imageSize - 2 * borderWidth, imageSize - 2 * borderWidth);

              currentImageIndex++;
            }
        }
    }

    drawImages(data);
//***************************************************************************** */

         const imageWidth = 128; // 图片宽度
         const imageSpacing = 20; // 图片间隔
         const startX = 50; // 起始 x 坐标
 
         // 创建一个空对象来存储图片分组
         const yGroupedImages = {};
 
         data.forEach(coord => {
         // 将 y 坐标相同的图片放入同一分组
             if (!yGroupedImages[coord[4]]) {
                 yGroupedImages[coord[4]] = [];
             }
             yGroupedImages[coord[4]].push(coord);
         });
 
         // 计算每一行的 x 坐标起点
         const xStartForRows = [startX, startX];
 
         let rowIndex = 0;
         
         let imagedowncx = [];
         let imagedowncy = [];
         let imagedowncc = [];
         let tem1 = 0
         let sum_x = 0

         for (const yCoord in yGroupedImages) {
            const imagesInGroup = yGroupedImages[yCoord];
            const totalImages = imagesInGroup.length;
        
            let selectedImages = [];
            if (totalImages <= 13) {
                selectedImages = imagesInGroup;
            } else {
                const step = totalImages / 13;
                for (let i = 0; i < 13; i++) {
                    const index = Math.floor(i * step);
                    selectedImages.push(imagesInGroup[index]);
                }
            }

            imagesInGroup.slice(0, 13).forEach((coord, index) => {
                if (sum_x < coord[1]){
                    sum_x = coord[1]
                }
            });
            }
        
        // 在合适的位置添加以下代码来绘制条形图*******************
       
        const groupedData = [];
        let currentGroup = [];
        let prevValue = null;
        const groupLengths = []; // 用于存储每个分组的长度
        let totalGroupCount = 0; // 用于累加分组中的数量
        const top_x_positions = []

        data.forEach(dataa => {
            const value = dataa[2];
            if (prevValue !== null && prevValue !== value) {
              groupedData.push(currentGroup);
              groupLengths.push(currentGroup.length); // 存储上一个分组的长度
              totalGroupCount += currentGroup.length; // 累加上一个分组的数量
              currentGroup = [];
            }
            currentGroup.push(dataa);
            prevValue = value;
          });
          groupedData.push(currentGroup); // 添加最后一个分组
          groupLengths.push(currentGroup.length); // 添加最后一个分组的长度
          totalGroupCount += currentGroup.length; // 累加最后一个分组的数量
          
          
          // 计算每个条形图的总值，用于计算宽度和间距
          const scale = 1980 / totalGroupCount;
          const yscale = 250 / Math.max(...groupLengths);
          
          // 绘制条形图
        //   let xPosition = 20;
          
        //   groupedData.forEach(group => {
        //     const totalValue = group.length;
        //     const heigt = yscale * totalValue;
        //     top_x_positions.push(xPosition);
          
        //     group.forEach(datab => {
        //       barChartGroup.append("rect")
        //         .attr("x", xPosition)
        //         .attr("y", 200 - datab[4] + 200) // 根据数量计算y坐标
        //         .attr("width", scale)
        //         .attr("height", heigt) // 使用数量作为高度
        //         .attr("fill", datab[3]) // 使用颜色数据
        //         .attr("fill-opacity", 0.7);
        //       xPosition += scale;
        //     });

            
          
        //   });
        //*********************** */

         //条形图的brush and points***********************************************************************************

         allpoint_positions = []
         

         data.forEach(allpointsdata =>{

            let point_x = allpointsdata[1] + 20;
            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", point_x);
            circle.setAttribute("cy", 325);
            circle.setAttribute("r", 7);
            circle.setAttribute("fill", allpointsdata[3]);
            circle.setAttribute("fill-opacity", 0.7);
            pointsGroup1.node().appendChild(circle);

            allpoint_positions.push(point_x);
            
         })

         
         const svg = d3.select("#line_arrow_svg"); // 选择已经存在的SVG元素
         // 创建一个包含brush的组
         
         const brushGroup = svg.append("g")
         .attr("class", "brush")
         .attr("transform", "translate(0, 300)"); // 调整brush的位置
         // 创建brush
         const brush_bar = d3.brushX()
         .extent([[10, 0], [2010, 50]]) // brush范围的坐标范围
         .on("end", brushed_bar); // 指定brush结束时的回调函数
         
         // 将brush添加到brush组
         
         brushGroup.call(brush_bar);

         function brushed_bar() {

            pointsGroup.selectAll("circle").remove();
            pointsGroup.selectAll("image").remove();
            pointsGroup.selectAll("rect").remove();
            d3.select('#line_arrow_svg').selectAll(".bezier").remove();
            d3.select("#Local_scatter").selectAll("svg").remove();

            const selection = d3.brushSelection(this); // 获取刷选范围
            const bar_topm_coorf = [];

            if (selection) {
                const [x0, x1] = selection; // 获取刷选的起始和结束坐标
                const selectedPointsf = [];
                const sel_xf = [];
                const index_listf = [];
                const selectedPoints = [];
                const sel_x = [];
                const index_list = [];
                const bar_topm_coor = [];

                allpoint_positions.forEach((point_x, index) => {
                    if (point_x >= x0 && point_x <= x1) {
                        selectedPointsf.push(data[index]); // 记录在刷选范围内的点的索引
                        bar_topm_coorf.push((scale*index) + (scale/2) + 20);
                        sel_xf.push(point_x);
                        index_listf.push(index);
                    }
                
                });
                const totallen = sel_xf.length;
                console.log(sel_xf, 111);
    
                if (totallen <= 13) {
                    for (let i = 0; i < totallen; i++){
                    selectedPoints.push(selectedPointsf[i]);
                    sel_x .push(sel_xf[i]);
                    index_list.push(index_listf[i]);
                    bar_topm_coor.push(bar_topm_coorf[i]);
                    }
                } else {
                    const step = totallen / 13;

                    for (let i = 0; i < 13; i++) {
                        const index = Math.floor(i * step);
                        selectedPoints.push(selectedPointsf[index]);
                        sel_x.push(sel_xf[index]);
                        index_list.push(index_listf[index]);
                        bar_topm_coor.push(bar_topm_coorf[index]);
                    }
                }
                console.log(sel_x);


                // allpoint_positions.forEach((point_x, index) => {
                //     if (point_x >= x0 && point_x <= x1) {
                //         selectedPoints.push(data[index]); // 记录在刷选范围内的点的索引
                //         bar_topm_coor.push((scale*index) + (scale/2) + 20);
                //         sel_x.push(point_x);
                //         index_list.push(index);
                //     }
                
                    
                // });



            // 打印刷选范围内点的坐标
            const sel_new_positions = [];
            const Min_sel_x = Math.min(...sel_x);
            const Max_sel_x = Math.max(...sel_x);
            const imagesInGroup = selectedPoints;
            sel_x.forEach(p_x => {
                sel_new_positions.push(((p_x - Min_sel_x)/(Max_sel_x - Min_sel_x))*1980);
            });



            let tem0 = 0;
            const imagedowncc = [];
            const positions = [];
            const imagedowncx = [];
            const circles = f_svg.selectAll("circle");
            
            for (const yCoord in yGroupedImages) {
                // const imagesInGroup = yGroupedImages[yCoord];
                // const imagesInGroup = yGroupedImages[yCoord];
                const totalImages = imagesInGroup.length;
                
            
                let selectedImages = [];
                if (totalImages <= 13) {
                    selectedImages = imagesInGroup;
                } else {
                    const step = totalImages / 13;
                    for (let i = 0; i < 13; i++) {
                        const index = Math.floor(i * step);
                        selectedImages.push(imagesInGroup[index]);
                    }
                }
                console.log(selectedImages, 1212)

    
                selectedImages.forEach((coord, index) => {
    
                    const x = xStartForRows[rowIndex] + index * (imageWidth + imageSpacing);
   
                    let point_x = (coord[1]/sum_x)*1980+10

                    let new_ind = index;
    
                    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle.setAttribute("cx", sel_new_positions[new_ind]+10);
                    circle.setAttribute("cy", 250);
                    circle.setAttribute("r", 10);
                    circle.setAttribute("fill", coord[3]);
                    circle.setAttribute("stroke", "black");
                    circle.setAttribute("stroke-width", 2);
                    circle.setAttribute("fill-opacity", 1);
                    pointsGroup.node().appendChild(circle);

                    const circle_ll = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                    circle_ll.setAttribute("cx", sel_x[new_ind]);
                    circle_ll.setAttribute("cy", 325);
                    circle_ll.setAttribute("r", 9);
                    circle_ll.setAttribute("stroke", "black");
                    circle_ll.setAttribute("stroke-width", 2);
                    circle_ll.setAttribute("fill", coord[3]);
                    circle_ll.setAttribute("fill-opacity", 0.7);
                    pointsGroup.node().appendChild(circle_ll);

                    circles
                    .filter((d, i) => i === coord[0])
                    .attr("r", 3.5)
                    .attr("fill-opacity", 0.7)
                    .attr("fill", coord[3]);
   
    
                    const image = document.createElementNS("http://www.w3.org/2000/svg", "image");
                    image.setAttribute("x", x);
                    image.setAttribute("y", coord[4]);
                    image.setAttribute("width", 128); // 设置图片宽度
                    image.setAttribute("height", 128); // 设置图片高度
                    const imagename = coord[0];
                    image.setAttribute("href", `cifar10/images256/cf10_image_${imagename}.jpg`); // 替换为实际图片的URL
                    pointsGroup.node().appendChild(image);

   
                    imagedowncx.push(x + 64)
                    if (coord[4] < 250) {
                       imagedowncy.push(coord[4] + 128)
                    } else {
                       imagedowncy.push(coord[4])
                    }
                   
                    imagedowncc.push(coord[3])
    
                    // 绘制带颜色的矩形（框）
                    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                    rect.setAttribute("x", x); // 根据图片位置调整矩形位置
                    rect.setAttribute("y", coord[4]); // 根据图片位置调整矩形位置
                    rect.setAttribute("width", 128); // 根据图片尺寸调整矩形尺寸
                    rect.setAttribute("height", 128); // 根据图片尺寸调整矩形尺寸
                    rect.setAttribute("stroke", coord[3]); // 设置框的颜色
                    rect.setAttribute("stroke-width", 5); // 设置框的线宽
                    rect.setAttribute("fill", "none"); // 不填充颜色，只显示边框
                    rect.setAttribute("fill-opacity", 0.7);
                    pointsGroup.node().appendChild(rect);
                    tem0++;
    
                });
                // rowIndex++;
                }
    
            // 绘制曲线
            var coordinates = [];
            var lit_coordinates = [];
            // var lit_bar_coordinates = [];
            
    
            //存储坐标
            let value = tem0
            const numDivs = value<=brush_ids.length?value:brush_ids.length;   // div的数目
            for(let i=0;i<numDivs;i++){
                //起始坐标
                
                var data1 =[{x:(sel_new_positions[i]+10), y:240},      // 这个position是连线上刻度的位置
                        {x:(imagedowncx[i]),y:imagedowncy[i]}];
                coordinates.push(data1);

                var data_little_points = [{x:(sel_x[i]), y:325},
                        {x:(sel_new_positions[i]+10), y:260}];
                lit_coordinates.push(data_little_points);

                // var data_little_points_bar = [{x:(sel_x[i]), y:325},
                //         {x:bar_topm_coor[i], y:395}];
                // lit_bar_coordinates.push(data_little_points_bar);
            }
    
            
            for(let i=0;i<numDivs;i++){
                drawBezier(coordinates[i], imagedowncc[i], 3);
                drawBezier(lit_coordinates[i], imagedowncc[i], 1);
                // drawBezier(lit_bar_coordinates[i], imagedowncc[i], 1);
            };

            highlightedIndices.splice(0, highlightedIndices.length);
            for (let i = 0; i < selectedPoints.length; i++) {
                if (i === 13) {
                    break;
                }
                highlightedIndices.push(selectedPoints[i][0]);
            }
            // const circles = f_svg.selectAll("circle");

            // const allIndices = [];  // 要高亮的点的索引
            // const allcolors = [];
            // for (let i = 0; i < data.length; i++) {
            //     allIndices.push(data[i][0]);
            //     allcolors.push(data[i][3]);
            // }

            data.forEach(coord => {

                circles
                .filter((d, i) => i === coord[0])
                .attr("r", 3.5)
                .attr("fill-opacity", 0.7)
                .attr("stroke-width", 0)
                .attr("fill", coord[3]);

                highlightedIndices.forEach(index => {
                    circles
                    .filter((d, i) => i === index)
                    .attr("fill-opacity", 0.9)
                    .attr("stroke", "black")  // 设置边框颜色为黑色
                    .attr("stroke-width", 2)
                    .attr("r", 10);  // 改变点的半径
                });


            });

            
            const dataURL1 = "https://gist.githubusercontent.com/Zehua-Yu/a1a43cb5bb7ceca68af18fce8be0e76f/raw/806a67664c7d87824ee64f82df26e958c62c3977/cifar10_tsne_data.csv"
 
 
            d3.csv(dataURL1, d3.autoType).then((data_ss) => {
            // 需要检查一下数据解析的结果，可能并不正确，需要在后面的步骤里再进行相应的处理
            console.log(data_ss);
            const new_data = [];
            selectedPoints.forEach(coord => {
                new_data.push(data_ss[coord[0]]);
            }); 
            
            
            global_data = new_data;
            
            const marginTop = 10; // top margin, in pixels
            const marginRight = 20; // right margin, in pixels
            const marginBottom = 30; // bottom margin, in pixels
            const marginLeft = 40; // left margin, in pixels
            // 构建一个单独的散点图
            const width=470;
            const height = 460;
            const svg2 = d3.select("#Local_scatter")
                .append("svg")
                .attr("x", 100).attr("y", 150)
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [-marginLeft, -marginTop, width, height]);
        
            // 设置绘制散点图所需的参数
            const scatterplotParams1 = {
                // 在原始数据中需要研究（映射）的属性
                x: "x", // x坐标
                y: "y", // y坐标
                z: (d) => d.color, // 种类通过颜色编码（映射）进行区分
                marginTop: marginTop,
                marginRight: marginRight,
                marginBottom: marginBottom,
                marginLeft: marginLeft,
                width: width,
                height: height
            };
        
            data.forEach((d)=>{
                if(d.x>max_x) max_x=d.x;
                if(d.x<min_x) min_x=d.x;
                if(d.y>max_y) max_y=d.y;
                if(d.y<min_y) min_y=d.y;
            })
            
            // 构建散点图
            ScatterplotMatrix_nobrush(new_data, svg2, scatterplotParams1);
        
        });         
    
            async function drawImages_bar(data, indexListToHighlight) {

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                let currentImageIndex = 0;
                for (let row = 0; row < numRows; row++) {
                    for (let col = 0; col < imagesPerRow; col++) {
                        if (currentImageIndex >= totalImages) {
                            return;
                        }
            
                        const x = col * (imageSize + spacing) + spacing;
                        const y = row * (imageSize + spacing) + spacing;
            
                        const imgSrc = `cifar10/images256/cf10_image_${data[currentImageIndex][0]}.jpg`;
                        const img = await loadImage(imgSrc);
            
                        // 绘制边框
                        
            
                        // 判断是否在索引列表中
                        const centerX = x + imageSize / 2;
                        const centerY = y + imageSize / 2;
                        const halfSize = 58 / 2;
                        const imgX = centerX - halfSize;
                        const imgY = centerY - halfSize;
                        ctx.drawImage(img, imgX, imgY, 58, 58);
                        ctx.strokeStyle = data[currentImageIndex][3];
                        ctx.lineWidth = borderWidth;
                        ctx.strokeRect(imgX, imgY, 58, 58);
                        ctx.fillRect(imgX, imgY, 58, 58);
                        ctx.fillStyle = 'rgba(211,211,211,0.7)';
                        if (indexListToHighlight.includes(currentImageIndex)) {
                            // 绘制图片
                            // ctx.drawImage(img, x + borderWidth, y + borderWidth, imageSize - 2 * borderWidth, imageSize - 2 * borderWidth);
                            // ctx.strokeStyle = data[currentImageIndex][3];
                            // ctx.lineWidth = borderWidth;
                            // ctx.strokeRect(x, y, imageSize, imageSize);
                            // ctx.fillStyle = 'white';
                            // ctx.fillRect(x, y, imageSize, imageSize);

                           //  ctx.fillStyle = 'rgba(255, 255, 80, 0.4)';
                            ctx.fillRect(x + borderWidth, y + borderWidth, imageSize - 2 * borderWidth, imageSize - 2 * borderWidth);
                            ctx.strokeStyle = "white";
                            ctx.lineWidth = borderWidth;
                            ctx.strokeRect(x, y, imageSize, imageSize);
            
                            // 绘制32x32的图像，以64x64的中心点为中心
                            const size1 = 73
                            const centerX = x + imageSize / 2;
                            const centerY = y + imageSize / 2;
                            const halfSize = size1 / 2;
                            const imgX = centerX - halfSize;
                            const imgY = centerY - halfSize;
                            ctx.drawImage(img, imgX, imgY, size1, size1);
                            ctx.strokeStyle = data[currentImageIndex][3];
                            ctx.lineWidth = borderWidth;
                            ctx.strokeRect(imgX, imgY, size1, size1);
                        } else {
                             // 绘制半透明阴影
                            //  ctx.fillStyle = 'white';
                            //  ctx.fillRect(x, y, imageSize, imageSize);
                            //  ctx.fillRect(x + borderWidth, y + borderWidth, imageSize - 2 * borderWidth, imageSize - 2 * borderWidth);
                            //  ctx.strokeStyle = "white";
                            //  ctx.lineWidth = borderWidth;
                            //  ctx.strokeRect(x, y, imageSize, imageSize);
             
                             // 绘制32x32的图像，以64x64的中心点为中心
                             const centerX = x + imageSize / 2;
                             const centerY = y + imageSize / 2;
                             const halfSize = 58 / 2;
                             const imgX = centerX - halfSize;
                             const imgY = centerY - halfSize;
                             ctx.drawImage(img, imgX, imgY, 58, 58);
                             ctx.strokeStyle = data[currentImageIndex][3];
                             ctx.lineWidth = borderWidth;
                             ctx.strokeRect(imgX, imgY, 58, 58);
                             ctx.fillRect(imgX, imgY, 58, 58);
                             ctx.fillStyle = 'rgba(211,211,211,0.7)';
             
                             
                        }
            
                        currentImageIndex++;
                    }
                }
        }
        console.log(index_list);
    
        drawImages_bar(data, index_list);

            }
          }
         //******************************************************************* */
 
         console.log('Response from server:', data);
     })
     .catch(error => {
         // 处理错误
         throw error;
     });
 
 }

 // 1. 定义高亮函数
function highlightPoints(selection) {
    selection
        .attr("r", 20)  // 改变点的半径
        .attr("fill", "black");  // 改变点的颜色
}


 
 function drawBezier(data,color, size_of_line){
 
     // 创建 SVG 路径生成器
     var lineGenerator = d3.line()
       .x(function(d) { return d.x; })   //提取每个点的 x和 y坐标
       .y(function(d) { return d.y; })
       .curve(d3.curveCardinal);     //.curve()方法设置了曲线的类型，这里使用的是d3.curveCardinal，表示使用Cardinal样条曲线
 
     // 计算控制点的坐标
     function getControlPoints(points) {
     //首先获取两个点的坐标，然后计算出两个控制点的坐标。
     //控制点的计算基于两个点之间的差值，并分别在两个点的 x轴上进行偏移。
       var p0 = points[0];
       var p1 = points[1];
       var dx = p1.x - p0.x;
       var dy = p1.y - p0.y;
       var cp1 = { x: p0.x + dx, y: p0.y+(dy/2) };
       var cp2 = { x: p1.x - dx, y: p1.y-(dy/2) };
       return [cp2, cp1];
     }
 
    //  // 绘制曲线
     var svg1 = d3.select('#line_arrow_svg'); // 选择 SVG 容器
    //  var svg = d3.select('svg');
     var controlPoints = getControlPoints(data);         //计算出控制点的坐标，并将结果保存在变量controlPoints中
     var pathData = "M " + data[0].x + " " + data[0].y;  //初始化为以第一个点的坐标为起点的SVG路径字符串
     var segment = [controlPoints[0], controlPoints[1], data[1]];
 
     // 将控制点和数据点的坐标通过.map()方法转换为字符串形式，并使用.join(", ")方法将它们连接起来
     // 这样就得到了表示贝塞尔曲线路径的一部分字符串。
     pathData += " C " + segment.map(function(d) { 
                                         return d.x + " " + d.y; 
                                     })
                                 .join(", ");
     

     svg1.append('path')
       .attr('d', pathData)
       .attr('fill', 'none')
       .attr('stroke', color)
       .attr('stroke-width', size_of_line)
       .attr("fill-opacity", 0.7)
       .attr('class', 'bezier')
       .lower();     //.lower()方法将绘制的曲线放置在SVG图层的最底部，以确保它在其他图形之下。

 }

//  function drawBezier(data, color, angleFactor) {
//     var svg1 = d3.select('#line_arrow_svg');
//     var controlPoints = getControlPoints(data, angleFactor); // Pass angleFactor to getControlPoints
//     var pathData = "M " + data[0].x + " " + data[0].y;
//     var segment = [controlPoints[0], controlPoints[1], data[1]];

//     pathData += " C " + segment.map(function(d) {
//         return d.x + " " + d.y;
//     }).join(", ");

//     svg1.append('path')
//         .attr('d', pathData)
//         .attr('fill', 'none')
//         .attr('stroke', color)
//         .attr('stroke-width', 3)
//         .attr("fill-opacity", 0.7)
//         .attr('class', 'bezier')
//         .lower();
// }

// function getControlPoints(points, angleFactor) {
//     var p0 = points[0];
//     var p1 = points[1];
//     var dx = p1.x - p0.x;
//     var dy = p1.y - p0.y;
//     var angle = Math.atan2(dy, dx);

//     var cpDistance = Math.sqrt(dx * dx + dy * dy) * angleFactor;
//     var cp1 = { x: p0.x + cpDistance * Math.cos(angle + Math.PI / 2), y: p0.y + cpDistance * Math.sin(angle + Math.PI / 2) };
//     var cp2 = { x: p1.x + cpDistance * Math.cos(angle - Math.PI / 2), y: p1.y + cpDistance * Math.sin(angle - Math.PI / 2) };

//     return [cp1, cp2];
// }
 
 function brush(cell, circle, svg, {data, padding, size, xScales, yScales, X, Y}) {
     const brush = d3.brush()
         .extent([[padding / 2, padding / 2], [size - padding / 2, size - padding / 2]])
         .on("start", brushstarted)
         .on("brush", brushed)
         .on("end", brushended);
   
     cell.call(brush);
    //  svg.selectAll("circle").remove();
    // //  .attr("r", 1)
    // //  .attr("stroke", 0)
    // //  .attr("fill-opacity", 0.7);
    //  ScatterplotMatrix(data, svg, scatterplotParams);
     
     let brushCell;
   
     // Clear the previously-active brush, if any.
     function brushstarted() {
                                      //  svg.selectAll("circle").remove();
                                    //   svg.selectAll("circle")
                                    //   .attr("r", 1)
                                    //   .attr("stroke", 0)
                                    //   .attr("fill-opacity", 0.7);
       
        const circles = svg.selectAll("circle");
        data.forEach(coord1 => {
            highlightedIndices.forEach(index => {
                circles
                .filter((d, i) => i === coord1["idx"])
                .attr("r", 3.5)
                .attr("fill-opacity", 0.7)
                .attr("stroke-width", 0)
                .attr("fill", coord1["color"]);
            });
        });
        
        if (brushCell !== this) {
         d3.select(brushCell).call(brush.move, null);
         brushCell = this;
       }
     }
   
     // Highlight the selected circles.
     function brushed({selection}, [x, y]) {
       let selected = [];
       if (selection) {
         const [[x0, y0], [x1, y1]] = selection; 
         circle.classed("hidden",
          (i) => x0 > xScales[x](X[x][i])
             || x1 < xScales[x](X[x][i])
             || y0 > yScales[y](Y[y][i])
             || y1 < yScales[y](Y[y][i]));
         selected = data.filter(
          (i) => x0 < xScales[x](X[x][i])
             && x1 > xScales[x](X[x][i])
             && y0 < yScales[y](Y[y][i])
             && y1 > yScales[y](Y[y][i]))
             .map((d) => d.idx); // 将选中的数据的索引值收集到数组中;
       }
       svg.property("value", selected).dispatch("input");
     }
   
     // If the brush is empty, select all circles.
     function brushended({selection}) {

         if (selection){
             const [[x0, y0], [x1, y1]] = selection;
             console.log("x0: ", x0, " y0 :", y0, "x1: ", x1, " y1 :", y1);
             
             data.forEach((d)=>{
                 const pos = brush_ids.indexOf(d.idx);
                 const dataX0 = xScales[0](d.x);
                 const dataY0 = yScales[0](d.y);
                 // console.log("dataX0: ", dataX0, " dataY0 :", dataY0);
                 if(dataX0 >= x0 && dataX0 <= x1 && dataY0 >= y0 && dataY0 <= y1){
                     if(pos == -1) brush_ids.push(d.idx);
                 }
                 else if(pos != -1) brush_ids.splice(pos, 1);
                 
             });
             selectedData = brush_ids;
             console.log("brush_ids: ", brush_ids);
             console.log("selectedData: ", selectedData);


        //  .attr("fill", data[]);

 
             send_brush_data(data, svg, url);
     
             return;
           };
       svg.property("value", []).dispatch("input");
       circle.classed("hidden", false);
     }
 }
 
 function ScatterplotMatrix_nobrush(
    data,
    svg,
    {
    // 一个数组，其元素是需要进行对比的属性名（或是 accessor function 属性的访问函数，从原始数据提取出该属性的值）
    // 默认是原始数据里的所有属性 data.columns
    // 根据需要对比的属性的数量 n，构建出一个相应的 n x n 散点图矩阵
    columns = data.columns,
    // 一个数组，矩阵横向的各个散点图的 x 轴所映射的属性（或 accessor function 访问函数）
    // 默认使用 columns 参数里的属性
    x = columns,
    y = columns, // 矩阵纵向的各个散点图的 y 轴所映射的属性
    // 数据点的 Z 轴的映射函数
    // 入参是各个数据点，返回相应的分类值
    // 默认都返回 1
    // 在该实例中，返回企鹅所属的种类进行分类
    z = () => 1,
    // 以下有一些关于图形的宽高、边距尺寸相关的参数
    padding = 20, // 矩阵中邻近「单元」（即散点图）之间的间隙（单位是像素）
    // 在外四边留白，构建一个显示的安全区，一般用于在四周显示坐标轴
    marginTop = 10, // top margin, in pixels
    marginRight = 10, // right margin, in pixels
    marginBottom = 30, // bottom margin, in pixels
    marginLeft = 30, // left margin, in pixels
    width = 500, // svg 的宽度
    height = width, // svg 的高度，默认和 width 宽度一样，因为构建的是一个正方形的 n x n 散点图矩阵
    // 因为该实例中所分析对比的各个属性的数据都是连续类型的，所以每一个散点图都可以采用相同的比例尺？
    xType = d3.scaleLinear, // 每一个散点图的横坐标轴所使用的比例尺类型，默认采用线性比例尺
    yType = d3.scaleLinear, // 每一个散点图的纵坐标轴所使用的比例尺类型
    zDomain, // 一个数组，Z 轴的定义域（即数据中的所有分类，对于该实例就是所有的企鹅种类）
    // 与数据点样式相关的参数
    fillOpacity = 0.7, // 数据点的透明度
    // 颜色 schema
    // 它将用于分类比例尺中
    // 将不同的数据分类（Z 轴的定义域）映射为不同的颜色）
    colors = ['#00BFFF', '#00FF00', '#FFA500', '#FF0000', '#800080', '#00FFFF', '#FF69B4', '#006400', '#A52A2A',
    '#00CED1'],
   //  colors = d3.schemeCategory10,
    x_range = [0,0], // 缩放用的参数
    y_range = [0,0]
    } = {}
) {
    /**
     *
     * 对原始数据 data 进行转换
     * 提取出相应的**轴**（X 轴、Y 轴，还可能包括其他维度，如 Z 轴，作为分类）所需的数据，以数组的形式
     * 然后在使用数据时，可以依据数据点的索引很方便地获取相应轴的值
     *
     */
    // 从原始数据 data 中提取出用于绘制矩阵中**每个散点图**的横坐标所需的数据
    // 由于该实例是 4 x 4 的散点图矩阵，所以 X 是一个具有 4 个元素的数组
    // 而 X 里面的每个元素还是数组（看上一个 cell 的演示）
    // 每个元素就是相应那一列的散点图的数据点的横坐标数据集
    // 所以在转换时需要进行两次转换 mapping，最终可以为每个散点图提取出其数据点的横坐标（对应一个属性）的值
    const X = d3.map(
    x, // 首先第一层映射是以矩阵横向的属性（数组）x 作为入参，这样 mapping 得到的是 4 个属性所对应的数据
    // 然后第二层映射是以原数据 data 作为入参，然后提取出相应的属性值
    (x) => d3.map(data, typeof x === "function" ? x : (d) => d[x])
    );
    // 从原始数据 data 中提取出用于绘制矩阵中**每个散点图**的纵坐标所需的数据
    // 该实例 Y 和 X 其实是一样的
    const Y = d3.map(y, (y) =>
    d3.map(data, typeof y === "function" ? y : (d) => d[y])
    );
    // 从原始数据 data 中提取数据点的分类依据的数据
    const Z = d3.map(data, z);

    /**
     *
     * 计算出 Z 轴的定义域，即数据中的所有分类（离散型数据）
     *
     */
    // 如果没有预设的种类 zDomain 则先将定义域设置为 Z，即先设置为所有数据点的分类依据值所构成的数据
    if (zDomain === undefined) zDomain = Z;
    // 然后再基于原来的 zDomain 值创建一个 InternSet 对象，以便去重（由于上面的 Z 中可能会有重复值，即使是指定了 zDomain 也未必能确保没有重复的分类值）
    // 这样所得的 zDomain 里的元素都是唯一的，作为 Z 轴的定义域（分类的依据）
    zDomain = new d3.InternSet(zDomain);

    // 在绘制散点图之前，这里还做了一步数据清洗
    // 使用 JavaScript 数组的原生方法 arr.filter() 筛掉不属于 zDomain 所预设的任何一类的数据点
    // 返回一个数组，其元素是一系列数字，对应于原数据集的元素的索引位置
    const I = d3.range(Z.length).filter((i) => zDomain.has(Z[i]));

    /**
     *
     * 设置矩阵「单元」（即散点图）的尺寸
     *
     */
    // 其中 X 和 Y 分别是前面数据转换得到的数组，其中 X.length 和 Y.length 长度就是矩阵在横向和纵向的维度（即在该方向上有几个属性，即散点图）
    const cellWidth =
    (width - marginLeft - marginRight - (X.length - 1) * padding) / X.length;
    const cellHeight =
    (height - marginTop - marginBottom - (Y.length - 1) * padding) / Y.length;
    let X_max = Math.max(...X[0]);
    let X_min = Math.min(...X[0]);
    let Y_max = Math.max(...Y[0]);
    let Y_min = Math.min(...Y[0]);
    let X_dis = Math.abs(X_max - X_min);
    let Y_dis = Math.abs(Y_max - Y_min);
    console.log(X_dis, Y_dis);


    /**
     *
     * 构建比例尺和坐标轴
     *
     */
    // 构建矩阵散点图的横向变量的（映射）比例尺
    // xScales 是一个数组，每一个元素对应于一列散点图的横轴（变量）比例尺
    var xScales = [];
    var yScales = [];
    if (X_dis < Y_dis) {
        xScales = X.map((X) => xType([X_min, X_min+Y_dis], [35, cellWidth-35]));
        yScales = Y.map((Y) => yType(d3.extent(Y), [cellHeight-35, 35]));
    }else {
        xScales = X.map((X) => xType(d3.extent(X), [35, cellWidth-35]));
        yScales = Y.map((Y) => yType([Y_min, Y_min+X_dis], [cellHeight-35, 35]));
    }
    
    // if(x_range[0]!==x_range[1]){
    //     xScales = X.map((X) => xType(x_range, [35, cellWidth-35]));
    // }
    
    // 构建矩阵散点图的纵向变量的（映射）比例尺
    // yScales 也是一个数组，每一个元素对应于一行散点图的纵轴（变量）比例尺
    // const yScales = Y.map((Y) => yType(d3.extent(Y), [cellHeight-35, 35]));
    // if(y_range[0]!==y_range[1]){
    //     yScales = Y.map((Y) => yType(y_range, [cellHeight-35, 35]));
    // }
    
    // 构建分类比例尺
    // 将离散的数据映射为不同的颜色
    const zScale = d3.scaleOrdinal(zDomain, colors);
    
    // 坐标轴对象
    const xAxis = d3.axisBottom().ticks(cellWidth / 50); // 横轴是一个朝下的坐标轴
    const yAxis = d3.axisLeft().ticks(cellHeight / 35); // 纵轴是一个朝左的坐标轴
  
    svg.append("style")
      .text(`circle.hidden { fill: #000; fill-opacity: 1; r: 1px; }`);

    // 绘制纵向坐标轴
    svg
    .append("g")
    .selectAll("g")
    // 绑定的数据是 yScales（不同散点图的纵向属性对应不同的比例尺）
    // 一个数组，每一个元素对应于一行散点图的纵轴（变量）比例尺
    .data(yScales)
    // 为每一行散点图创建一个纵向坐标轴容器
    .join("g")
    // 通过设置 CSS 的 transform 属性将这些纵向坐标轴容器「移动」到相应的位置
    // 其中 translate() 第一个参数 `0` 表示将每一个纵坐标轴容器都定位到左侧
    // 而第二个参数 ${i * (cellHeight + padding)} 表示每一个纵坐标轴容器的高度会随着其绑定的数据的索引值而变化
    .attr("transform", (d, i) => `translate(0, ${i * (cellHeight + padding)})`)
    // 对选择集中的每个元素（纵向坐标轴容器）都调用一次函数 function 执行相应的操作
    // 该函数的入参是纵向坐标轴容器所绑定的数据（该属性相应的比例尺）
    .each(function (yScale) {
        // 在纵向坐标轴容器里用相应的比例尺绘制出坐标轴
        // 在函数内的 this 是指当前迭代的纵向坐标轴容器 <g> 元素
        return d3.select(this).call(yAxis.scale(yScale));
    })
    .call((g) => g.select(".domain").remove()) // 删掉上一步所生成的坐标轴的轴线（它含有 domain 类名）
    .call((g) =>
        g
        .selectAll(".tick line")
        .clone() // 这里复制了一份刻度线，用以绘制散点图中纵向的网格参考线
        .attr("x2", width - marginLeft - marginRight) // 调整复制后的刻度线的终点位置（往右移动）
        .attr("stroke-opacity", 0.1)
    ); // 调小网格线的透明度

    // 绘制横向坐标轴
    svg
    .append("g")
    .selectAll("g")
    .data(xScales)
    .join("g")
    // 通过设置 CSS 的 transform 属性将这些横向坐标轴容器「移动」到相应的位置
    // 其中 translate() 第一个参数 ${i * (cellWidth + padding)} 表示每一个横坐标轴容器的水平位置会随着其绑定的数据的索引值而变化
    // 而第二个参数 ${height - marginBottom - marginTop} 表示将每一个横坐标轴容器都定位到底部
    .attr(
        "transform",
        (d, i) =>
        `translate(${i * (cellWidth + padding)}, ${height - marginBottom - marginTop
        })`
    )
    .each(function (xScale) {
        return d3.select(this).call(xAxis.scale(xScale));
    })
    .call((g) => g.select(".domain").remove())
    .call((g) =>
        g
        .selectAll(".tick line")
        .clone()
        .attr("y2", -height + marginTop + marginBottom)
        .attr("stroke-opacity", 0.1)
    );

    /**
     *
     * 构建散点图矩阵的各个「单元」
     *
     */
    const cell = svg
    .append("g")
    .selectAll("g")
    // 基于行和列的维度，构建出索引值作为矩阵中各个「单元」所绑定的数据
    // 使用 d3.range() 生成一个等差数列，作为行/列的索引值
    // 使用 d3.cross() 将两个数组的元素交叉组合 [i, j] 作为二维矩阵中各元素的索引值
    // 例如 [1, 1] 可以表示第一行第一列的那个散点图
    .data(d3.cross(d3.range(X.length), d3.range(Y.length)))
    .join("g")
    .attr("fill-opacity", fillOpacity) // 设置透明度
    // 通过设置 CSS 的 transform 属性，基于每个「单元」所绑定的索引值，将它们「移动」到相应的（行和列）位置
    // 其中 translate() 第一个参数 ${i * (cellWidth + padding)} 表示该「单元」的横向位置
    // 而第二个参数 ${j * (cellHeight + padding)} 表示该「单元」的纵向位置
    .attr(
        "transform",
        ([i, j]) =>
        `translate(${i * (cellWidth + padding)}, ${j * (cellHeight + padding)})`
    );

    // 为每个「单元」设置一个边框，以便区分邻近的散点图
    cell
    .append("rect")
    .attr("fill", "none")
    .attr("stroke", "currentColor")
    .attr("width", cellWidth)
    .attr("height", cellHeight);

    // 绘制数据点
    // 对每个「单元」（散点图容器）都调用一次函数 function 执行相应的操作
    // 该函数的入参是每个「单元」（散点图容器）所绑定的数据（即「单元」所对应的索引值 [x, y]，一个二元数组）


    cell.each(function ([x, y]) {
  // 在函数内的 this 是指当前迭代的「单元」（散点图容器） <g> 元素
  // 将数据点绘制在散点图中
  d3.select(this)
  .selectAll("circle")
  // 这里在绑定数据时，再进行一次数据清洗
  // （基于 i 索引值进行迭代）筛掉在当前散点图所对应的横向属性 X[x] 或纵向属性 Y[y] 任意一个为空的数据点
  // 即 !isNaN(X[x][i]) 和 isNaN(Y[y][i]) 均需要成立
  .data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
  .join("circle")
  // 设置数据点的大小（圆的半径大小）
  // .attr("r", 3.5)
  // 设置各个 <circle> 元素的属性 cx 和 cy 将其移动到相应的位置
  // 其中 X[x][i] 就是当前数据点的横向（原始）值，xScales[x] 就是当前散点图的比例尺（用于对数据进行映射）
  .attr("cx", (i) =>  xScales[x](X[x][i]))
  .attr("cy", (i) =>  yScales[y](Y[y][i])); // 纵向值
//   .attr("cx", (i) => {
//     if (X_dis > Y_dis) {
//         return xScales[x](X[x][i]);
//     }
//     return yScales[x](X[x][i]);
//   })
//   .attr("cy", (i) => {    
//     if (X_dis > Y_dis) {
//         return xScales[y](Y[y][i]);
//     }
//     return yScales[y](Y[y][i]);}); // 纵向值
// 添加连线

svg.append("defs")
  .append("marker")
  .attr("id", "arrow")
  .attr("viewBox", "0 -5 10 10")
  .attr("refX", 10) // 箭头标记的偏移位置
  .attr("markerWidth", 10) // 箭头标记的宽度
  .attr("markerHeight", 8) // 箭头标记的高度
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-5L10,0L0,5") // 箭头的形状
  .style("fill", "black"); // 箭头的颜色
  // 添加连线
d3.select(this)
.selectAll("line")
.data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
.join("line")
.attr("x1", (i, idx, nodes) => {
  if (idx < nodes.length - 1) {
      return xScales[x](X[x][idx]);
    }
    return null; // 最后一个点不绘制线段
})
.attr("y1", (i, idx, nodes) => {
  if (idx < nodes.length - 1) {
    return yScales[y](Y[y][idx]);
  }
  return null; // 最后一个点不绘制线段
})
.attr("x2", (i, idx, nodes) => {
  if (idx < nodes.length - 1) {
    return xScales[x](X[x][idx + 1]);
  }
  return null; // 最后一个点不绘制线段
})
.attr("y2", (i, idx, nodes) => {
  if (idx < nodes.length - 1) {
    return yScales[y](Y[y][idx + 1]);
  }
  return null; // 最后一个点不绘制线段
})
.attr("stroke", "gray")
.attr("stroke-width", 1);
// .attr("marker-end", (i, idx, nodes) => {
//     if (idx < nodes.length - 1) {
//       return "url(#arrow)";
//     }
//     return null; // 最后一个点不绘制线段
//   }); // 应用箭头标记

// d3.select(this)
//   .selectAll("line")
//   .data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
//   .join("line")
//   .attr("x1", (i, idx, nodes) => {
//     if (idx < nodes.length - 1) {
//         return xScales[x](X[x][idx]);
//       }
//       return null; // 最后一个点不绘制线段
//   })
//   .attr("y1", (i, idx, nodes) => {
//     if (idx < nodes.length - 1) {
//       return yScales[y](Y[y][idx]);
//     }
//     return null; // 最后一个点不绘制线段
//   })
//   .attr("x2", (i, idx, nodes) => {
//     if (idx < nodes.length - 1) {
//       return xScales[x](X[x][idx + 1]);
//     }
//     return null; // 最后一个点不绘制线段
//   })
//   .attr("y2", (i, idx, nodes) => {
//     if (idx < nodes.length - 1) {
//       return yScales[y](Y[y][idx + 1]);
//     }
//     return null; // 最后一个点不绘制线段
//   })
//   .attr("stroke", "gray")
//   .attr("stroke-width", 1);
});

    // cell.each(function ([x, y]) {

    // // 在函数内的 this 是指当前迭代的「单元」（散点图容器） <g> 元素
    // // 将数据点绘制在散点图中
    //     d3.select(this)
    //         .selectAll("circle")
    //         // 这里在绑定数据时，再进行一次数据清洗
    //         // （基于 i 索引值进行迭代）筛掉在当前散点图所对应的横向属性 X[x] 或纵向属性 Y[y] 任意一个为空的数据点
    //         // 即 !isNaN(X[x][i]) 和 isNaN(Y[y][i]) 均需要成立
    //         .data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
    //         .join("circle")
    //         // 设置数据点的大小（圆的半径大小）
    //         // .attr("r", 3.5) 
    //         // 设置各个 <circle> 元素的属性 cx 和 cy 将其移动到相应的位置
    //         // 其中 X[x][i] 就是当前数据点的横向（原始）值，xScales[x] 就是当前散点图的比例尺（用于对数据进行映射）
    //         .attr("cx", (i) => xScales[x](X[x][i]))
    //         .attr("cy", (i) => yScales[y](Y[y][i])); // 纵向值
    //         // 添加连线
    //         d3.select(this)
    //             .selectAll("line")
    //             .data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
    //             .join("line")
    //             .attr("x1", (i) => xScales[x](X[x][i]))
    //             .attr("y1", (i) => yScales[y](Y[y][i]))
    //             .attr("x2", (i, idx, nodes) => {
    //                 if (idx <= nodes.length) {
    //                     return xScales[x](X[x][idx + 1]);
    //                 }
    //                 return null; // 最后一个点不绘制线段
    //             })
    //             .attr("y2", (i, idx, nodes) => {
    //                 if (idx <= nodes.length) {
    //                     return yScales[y](Y[y][idx + 1]);
    //                 }
    //                 return null; // 最后一个点不绘制线段
    //             })
    //             .attr("stroke", "gray")
    //             .attr("stroke-width", 1);
    //        //  .attr("fill", (i) => zScale(Z[i])); // 设置数据点的颜色，根据 Z 比例尺来设定
           
    // });

    const circle = cell.selectAll("circle")
      .attr("r", 10)
      .attr("stroke", "black")
      .attr("stroke-width", 2)
      .attr("fill-opacity", 0.7)
      .attr("fill", (i) => Z[i]);
    
   //    .attr("fill", (i) => zScale(Z[i]));

    return Object.assign(svg.node(), { scales: { color: zScale } });
}

 function ScatterplotMatrix(
     data,
     svg,
     {
     // 一个数组，其元素是需要进行对比的属性名（或是 accessor function 属性的访问函数，从原始数据提取出该属性的值）
     // 默认是原始数据里的所有属性 data.columns
     // 根据需要对比的属性的数量 n，构建出一个相应的 n x n 散点图矩阵
     columns = data.columns,
     // 一个数组，矩阵横向的各个散点图的 x 轴所映射的属性（或 accessor function 访问函数）
     // 默认使用 columns 参数里的属性
     x = columns,
     y = columns, // 矩阵纵向的各个散点图的 y 轴所映射的属性
     // 数据点的 Z 轴的映射函数
     // 入参是各个数据点，返回相应的分类值
     // 默认都返回 1
     // 在该实例中，返回企鹅所属的种类进行分类
     z = () => 1,
     // 以下有一些关于图形的宽高、边距尺寸相关的参数
     padding = 20, // 矩阵中邻近「单元」（即散点图）之间的间隙（单位是像素）
     // 在外四边留白，构建一个显示的安全区，一般用于在四周显示坐标轴
     marginTop = 10, // top margin, in pixels
     marginRight = 10, // right margin, in pixels
     marginBottom = 30, // bottom margin, in pixels
     marginLeft = 30, // left margin, in pixels
     width = 928, // svg 的宽度
     height = width, // svg 的高度，默认和 width 宽度一样，因为构建的是一个正方形的 n x n 散点图矩阵
     // 因为该实例中所分析对比的各个属性的数据都是连续类型的，所以每一个散点图都可以采用相同的比例尺？
     xType = d3.scaleLinear, // 每一个散点图的横坐标轴所使用的比例尺类型，默认采用线性比例尺
     yType = d3.scaleLinear, // 每一个散点图的纵坐标轴所使用的比例尺类型
     zDomain, // 一个数组，Z 轴的定义域（即数据中的所有分类，对于该实例就是所有的企鹅种类）
     // 与数据点样式相关的参数
     fillOpacity = 0.7, // 数据点的透明度
     // 颜色 schema
     // 它将用于分类比例尺中
     // 将不同的数据分类（Z 轴的定义域）映射为不同的颜色）
     colors = ['#00BFFF', '#00FF00', '#FFA500', '#FF0000', '#800080', '#00FFFF', '#FF69B4', '#006400', '#A52A2A',
     '#00CED1'],
    //  colors = d3.schemeCategory10,
     x_range = [0,0], // 缩放用的参数
     y_range = [0,0]
     } = {}
 ) {

    // d3.select("#barChartGroup").select(".brush").call(brush.move, null);

     /**
      *
      * 对原始数据 data 进行转换
      * 提取出相应的**轴**（X 轴、Y 轴，还可能包括其他维度，如 Z 轴，作为分类）所需的数据，以数组的形式
      * 然后在使用数据时，可以依据数据点的索引很方便地获取相应轴的值
      *
      */
     // 从原始数据 data 中提取出用于绘制矩阵中**每个散点图**的横坐标所需的数据
     // 由于该实例是 4 x 4 的散点图矩阵，所以 X 是一个具有 4 个元素的数组
     // 而 X 里面的每个元素还是数组（看上一个 cell 的演示）
     // 每个元素就是相应那一列的散点图的数据点的横坐标数据集
     // 所以在转换时需要进行两次转换 mapping，最终可以为每个散点图提取出其数据点的横坐标（对应一个属性）的值
     const X = d3.map(
     x, // 首先第一层映射是以矩阵横向的属性（数组）x 作为入参，这样 mapping 得到的是 4 个属性所对应的数据
     // 然后第二层映射是以原数据 data 作为入参，然后提取出相应的属性值
     (x) => d3.map(data, typeof x === "function" ? x : (d) => d[x])
     );
     // 从原始数据 data 中提取出用于绘制矩阵中**每个散点图**的纵坐标所需的数据
     // 该实例 Y 和 X 其实是一样的
     const Y = d3.map(y, (y) =>
     d3.map(data, typeof y === "function" ? y : (d) => d[y])
     );
     // 从原始数据 data 中提取数据点的分类依据的数据
     const Z = d3.map(data, z);
 
     /**
      *
      * 计算出 Z 轴的定义域，即数据中的所有分类（离散型数据）
      *
      */
     // 如果没有预设的种类 zDomain 则先将定义域设置为 Z，即先设置为所有数据点的分类依据值所构成的数据
     if (zDomain === undefined) zDomain = Z;
     // 然后再基于原来的 zDomain 值创建一个 InternSet 对象，以便去重（由于上面的 Z 中可能会有重复值，即使是指定了 zDomain 也未必能确保没有重复的分类值）
     // 这样所得的 zDomain 里的元素都是唯一的，作为 Z 轴的定义域（分类的依据）
     zDomain = new d3.InternSet(zDomain);
 
     // 在绘制散点图之前，这里还做了一步数据清洗
     // 使用 JavaScript 数组的原生方法 arr.filter() 筛掉不属于 zDomain 所预设的任何一类的数据点
     // 返回一个数组，其元素是一系列数字，对应于原数据集的元素的索引位置
     const I = d3.range(Z.length).filter((i) => zDomain.has(Z[i]));
 
     /**
      *
      * 设置矩阵「单元」（即散点图）的尺寸
      *
      */
     // 其中 X 和 Y 分别是前面数据转换得到的数组，其中 X.length 和 Y.length 长度就是矩阵在横向和纵向的维度（即在该方向上有几个属性，即散点图）
     const cellWidth =
     (width - marginLeft - marginRight - (X.length - 1) * padding) / X.length;
     const cellHeight =
     (height - marginTop - marginBottom - (Y.length - 1) * padding) / Y.length;
 
     /**
      *
      * 构建比例尺和坐标轴
      *
      */
     // 构建矩阵散点图的横向变量的（映射）比例尺
     // xScales 是一个数组，每一个元素对应于一列散点图的横轴（变量）比例尺
 
     const xScales = X.map((X) => xType(d3.extent(X), [10, cellWidth-10]));
     if(x_range[0]!==x_range[1]){
         xScales = X.map((X) => xType(x_range, [10, cellWidth-10]));
     }
     
     // 构建矩阵散点图的纵向变量的（映射）比例尺
     // yScales 也是一个数组，每一个元素对应于一行散点图的纵轴（变量）比例尺
     const yScales = Y.map((Y) => yType(d3.extent(Y), [cellHeight-10, 10]));
     if(y_range[0]!==y_range[1]){
         yScales = Y.map((Y) => yType(y_range, [cellHeight-10, 10]));
     }
     
     // 构建分类比例尺
     // 将离散的数据映射为不同的颜色
     const zScale = d3.scaleOrdinal(zDomain, colors);
     
     // 坐标轴对象
     const xAxis = d3.axisBottom().ticks(cellWidth / 50); // 横轴是一个朝下的坐标轴
     const yAxis = d3.axisLeft().ticks(cellHeight / 35); // 纵轴是一个朝左的坐标轴
   
     svg.append("style")
       .text(`circle.hidden { fill: #000; fill-opacity: 1; r: 1px; }`);
 
     // 绘制纵向坐标轴
     svg
     .append("g")
     .selectAll("g")
     // 绑定的数据是 yScales（不同散点图的纵向属性对应不同的比例尺）
     // 一个数组，每一个元素对应于一行散点图的纵轴（变量）比例尺
     .data(yScales)
     // 为每一行散点图创建一个纵向坐标轴容器
     .join("g")
     // 通过设置 CSS 的 transform 属性将这些纵向坐标轴容器「移动」到相应的位置
     // 其中 translate() 第一个参数 `0` 表示将每一个纵坐标轴容器都定位到左侧
     // 而第二个参数 ${i * (cellHeight + padding)} 表示每一个纵坐标轴容器的高度会随着其绑定的数据的索引值而变化
     .attr("transform", (d, i) => `translate(0, ${i * (cellHeight + padding)})`)
     // 对选择集中的每个元素（纵向坐标轴容器）都调用一次函数 function 执行相应的操作
     // 该函数的入参是纵向坐标轴容器所绑定的数据（该属性相应的比例尺）
     .each(function (yScale) {
         // 在纵向坐标轴容器里用相应的比例尺绘制出坐标轴
         // 在函数内的 this 是指当前迭代的纵向坐标轴容器 <g> 元素
         return d3.select(this).call(yAxis.scale(yScale));
     })
     .call((g) => g.select(".domain").remove()) // 删掉上一步所生成的坐标轴的轴线（它含有 domain 类名）
     .call((g) =>
         g
         .selectAll(".tick line")
         .clone() // 这里复制了一份刻度线，用以绘制散点图中纵向的网格参考线
         .attr("x2", width - marginLeft - marginRight) // 调整复制后的刻度线的终点位置（往右移动）
         .attr("stroke-opacity", 0.1)
     ); // 调小网格线的透明度
 
     // 绘制横向坐标轴
     svg
     .append("g")
     .selectAll("g")
     .data(xScales)
     .join("g")
     // 通过设置 CSS 的 transform 属性将这些横向坐标轴容器「移动」到相应的位置
     // 其中 translate() 第一个参数 ${i * (cellWidth + padding)} 表示每一个横坐标轴容器的水平位置会随着其绑定的数据的索引值而变化
     // 而第二个参数 ${height - marginBottom - marginTop} 表示将每一个横坐标轴容器都定位到底部
     .attr(
         "transform",
         (d, i) =>
         `translate(${i * (cellWidth + padding)}, ${height - marginBottom - marginTop
         })`
     )
     .each(function (xScale) {
         return d3.select(this).call(xAxis.scale(xScale));
     })
     .call((g) => g.select(".domain").remove())
     .call((g) =>
         g
         .selectAll(".tick line")
         .clone()
         .attr("y2", -height + marginTop + marginBottom)
         .attr("stroke-opacity", 0.1)
     );
 
     /**
      *
      * 构建散点图矩阵的各个「单元」
      *
      */
     const cell = svg
     .append("g")
     .selectAll("g")
     // 基于行和列的维度，构建出索引值作为矩阵中各个「单元」所绑定的数据
     // 使用 d3.range() 生成一个等差数列，作为行/列的索引值
     // 使用 d3.cross() 将两个数组的元素交叉组合 [i, j] 作为二维矩阵中各元素的索引值
     // 例如 [1, 1] 可以表示第一行第一列的那个散点图
     .data(d3.cross(d3.range(X.length), d3.range(Y.length)))
     .join("g")
     .attr("fill-opacity", fillOpacity) // 设置透明度
     // 通过设置 CSS 的 transform 属性，基于每个「单元」所绑定的索引值，将它们「移动」到相应的（行和列）位置
     // 其中 translate() 第一个参数 ${i * (cellWidth + padding)} 表示该「单元」的横向位置
     // 而第二个参数 ${j * (cellHeight + padding)} 表示该「单元」的纵向位置
     .attr(
         "transform",
         ([i, j]) =>
         `translate(${i * (cellWidth + padding)}, ${j * (cellHeight + padding)})`
     );
 
     // 为每个「单元」设置一个边框，以便区分邻近的散点图
     cell
     .append("rect")
     .attr("fill", "none")
     .attr("stroke", "currentColor")
     .attr("width", cellWidth)
     .attr("height", cellHeight);
 
     // 绘制数据点
     // 对每个「单元」（散点图容器）都调用一次函数 function 执行相应的操作
     // 该函数的入参是每个「单元」（散点图容器）所绑定的数据（即「单元」所对应的索引值 [x, y]，一个二元数组）
     cell.each(function ([x, y]) {

     // 在函数内的 this 是指当前迭代的「单元」（散点图容器） <g> 元素
     // 将数据点绘制在散点图中
         d3.select(this)
             .selectAll("circle")
             // 这里在绑定数据时，再进行一次数据清洗
             // （基于 i 索引值进行迭代）筛掉在当前散点图所对应的横向属性 X[x] 或纵向属性 Y[y] 任意一个为空的数据点
             // 即 !isNaN(X[x][i]) 和 isNaN(Y[y][i]) 均需要成立
             .data(I.filter((i) => !isNaN(X[x][i]) && !isNaN(Y[y][i])))
             .join("circle")
             // 设置数据点的大小（圆的半径大小）
             // .attr("r", 3.5) 
             // 设置各个 <circle> 元素的属性 cx 和 cy 将其移动到相应的位置
             // 其中 X[x][i] 就是当前数据点的横向（原始）值，xScales[x] 就是当前散点图的比例尺（用于对数据进行映射）
             .attr("cx", (i) => xScales[x](X[x][i]))
             .attr("cy", (i) => yScales[y](Y[y][i])); // 纵向值
            //  .attr("fill", (i) => zScale(Z[i])); // 设置数据点的颜色，根据 Z 比例尺来设定
     });
 
     const circle = cell.selectAll("circle")
       .attr("r", 3.5)
       .attr("fill-opacity", 0.7)
       .attr("fill", (i) => Z[i]);
    //    .attr("fill", (i) => zScale(Z[i]));
 
     const size = cellWidth + padding;
 
     // Ignore this line if you don't need the brushing behavior.


     selected = cell.call(brush, circle, svg, {data, padding, size, xScales, yScales, X, Y});
 
     // 当散点图矩阵（横向维度等于纵向维度时 x===y）是一个方阵时（TODO 是否需要支持在非对称的散点图矩阵也添加标注？）
     // 在对角线上的散点图添加标注
     // 以表示（在对角线上）散点图所在的行和列所表示的属性变量
     if (x === y)
     svg
         .append("g") // 创建标注文本的容器
         .attr("font-size", 10)
         .attr("font-family", "sans-serif")
         .attr("font-weight", "bold")
         .selectAll("text")
         // 因为这是一个方阵，横轴和纵轴的散点图所映射的变量数量和名称都是相同的
         // 所以在绑定数据时，只需要绑定横向轴（或纵向轴）所映射的属性即可
         .data(x)
         .join("text")
         // 通过设置 CSS 的 transform 属性，将它们「移动」到相应的位置
         // 基于每个标注文本的容器所绑定的索引值，将它们定位到相应的（对角线上的）散点图里
         .attr(
         "transform",
         (d, i) =>
             `translate(${i * (cellWidth + padding)}, ${i * (cellHeight + padding)
             })`
         )
         // 为标注文本设置定位（相对于其容器）和纵向的偏移 dy，避免文字贴着散点图的边框
         .attr("x", padding / 2)
         .attr("y", padding / 2)
         .attr("dy", ".71em")
         .text((d) => d); // 设置标注内容
 
     return Object.assign(svg.node(), { scales: { color: zScale } });
 }
 
 /**
  *
  * 构建 svg 并获取尺寸大小等参数
  *
  */
 const container = document.getElementById("container"); // 图像的容器
 
 // 获取尺寸大小
 const windowWidth = document.documentElement.clientWidth; // 页面宽度
 const windowHeight = document.documentElement.clientHeight; // 页面高度
 buttonValue.addEventListener("change", function(){
    url = url_change();
 })

 console.log(url);
 console.log("windowWidth:",windowWidth);
 console.log("windowHeight:",windowHeight);
 //   let width = windowWidth;
 //   let height = windowHeight * 0.9;
 
 //   // 由于绘制的散点图矩阵是一个方阵
 //   // 所以要将 svg 的宽高设置为一样（以较小的一个边作为基准）
 //   if (width > height) {
 //     width = height;
 //   } else {
 //     height = width;
 //   }
 let width = 515;
 let height = 515;
 
 console.log({ width, height });
 
 const marginTop = 10; // top margin, in pixels
 const marginRight = 20; // right margin, in pixels
 const marginBottom = 30; // bottom margin, in pixels
 const marginLeft = 40; // left margin, in pixels
 // 创建 svg
 // 在容器 <div id="container"> 元素内创建一个 SVG 元素
 // 返回一个选择集，只有 svg 一个元素
 // const svg_m = d3
 //     .select("#container")
 //     .append("svg")
 //     .attr("x", 100).attr("y", 150)
 //     .attr("width", width)
 //     .attr("height", height)
 //     .attr("viewBox", [-marginLeft, -marginTop, width, height]);
 
 /**
  *
  * 异步获取数据
  * 再在回调函数中执行绘制操作
  *
  */
 
 // const dataURL1 =
 //     "https://gist.githubusercontent.com/FFFFFancy/b40d335fb9be0e9e248ce03d6002dc87/raw/b92ac0f39b8cbb02b6a0ca101dee5b00ffbd1d96/datad0.csv";

 function url_change(){
    // 获取loss的相关信息
    

    const getValue = (button) => {
       return new Promise((resolve, reject) => {
           button.addEventListener("change", (event) => {
               resolve(event.target.value);
           });
       });
    };
   
    // let url;
    getValue(buttonValue).then((lossValue) => {
       const type = JSON.stringify(lossValue);
       console.log("button:", lossValue);
       console.log("type:", typeof(lossValue));
       console.log("buttontype:", lossValue == "TSP");

       if(lossValue == "TSP"){
           url = path+'/api/data/TSP';
       }else if(lossValue == "Stress Majorization"){
           url = path+'/api/data/SM';
       }else if(lossValue == "Moran's I"){
           url = path+'/api/data/MI';
       }else{
           url = path+'/api/data/TSP';
       };
       console.log(url);
    });
    return url;
 }
 
 const dataURL1 = "https://gist.githubusercontent.com/Zehua-Yu/a1a43cb5bb7ceca68af18fce8be0e76f/raw/806a67664c7d87824ee64f82df26e958c62c3977/cifar10_tsne_data.csv"
 
 
     d3.csv(dataURL1, d3.autoType).then((data) => {
    

     // 需要检查一下数据解析的结果，可能并不正确，需要在后面的步骤里再进行相应的处理
     global_data = data;
     // 构建一个单独的散点图
     const svg1 = d3.select("#container")
         .append("svg")
         .attr("x", 100).attr("y", 150)
         .attr("width", width)
         .attr("height", height)
         .attr("viewBox", [-marginLeft, -marginTop, width, height]);
 
     // 设置绘制散点图所需的参数
     const scatterplotParams = {
         // 在原始数据中需要研究（映射）的属性
         x: "x", // x坐标
         y: "y", // y坐标
         z: (d) => d.color, // 种类通过颜色编码（映射）进行区分
         marginTop: marginTop,
         marginRight: marginRight,
         marginBottom: marginBottom,
         marginLeft: marginLeft,
         width: width,
         height: height
     };
 
     data.forEach((d)=>{
         if(d.x>max_x) max_x=d.x;
         if(d.x<min_x) min_x=d.x;
         if(d.y>max_y) max_y=d.y;
         if(d.y<min_y) min_y=d.y;
     })

     

     
     // 构建散点图
     ScatterplotMatrix(data, svg1, scatterplotParams);
 
 });