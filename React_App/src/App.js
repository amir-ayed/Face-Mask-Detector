import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";


function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const blazeface = require('@tensorflow-models/blazeface');

  // Main function
  const runCoco = async () => {
    const net = await tf.loadLayersModel('http://localhost/tfjs_model2/model.json');
    const facenet = await blazeface.load();

    //  Loop and detect
    setInterval(() => {
      detect(net, facenet);
    }, 500);
  };

  const detect = async (net, facenet) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const predictions = await facenet.estimateFaces(video, true, false, false);

      const ctx = canvasRef.current.getContext("2d");

      if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
          var text = ""
          var start = predictions[i].topLeft.arraySync();
          var end = predictions[i].bottomRight.arraySync();
          var size = [end[0] - start[0], end[1] - start[1]];
          if(videoWidth<end[0] && videoHeight<end[0]){
			    	console.log("image out of frame")
			    	continue
			    }

          var inputImage = tf.browser.fromPixels(video).toFloat()
			    inputImage=inputImage.slice([parseInt(start[1])+10,parseInt(start[0]),0],[parseInt(size[1]),parseInt(size[0]),3])
			    inputImage=inputImage.resizeBilinear([224,224]).reshape([1,224,224,3])
			    const result=net.predict(inputImage).dataSync()

			    if (result[1]>result[0]){
			    	//no mask on
              ctx.lineWidth = "3";
			      	ctx.strokeStyle="red"
			      	ctx.fillStyle = "red";
			      	text = "No Mask: "+(result[1]*100).toFixed(2).toString()+"%";
			    }else{
			    	//mask on
              ctx.lineWidth = "3";
			      	ctx.strokeStyle="green"
			      	ctx.fillStyle = "green";
			      	text = "Mask: "+(result[0]*100).toFixed(2).toString()+"%";
			    }

          ctx.beginPath();
          ctx.rect(start[0], start[1]+15, size[0], size[1]);
          ctx.stroke()
          ctx.font = "bold 12pt sans-serif";
			    ctx.fillText(text,start[0]+5,start[1]-5)
        }
      }
    }
  };

  useEffect(()=>{runCoco()},[]);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          muted={true} 
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 8,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
