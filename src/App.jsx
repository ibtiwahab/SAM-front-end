import React, { useState } from "react";
import CanvasSegmentor from "./components/CanvasSegmentor";
import "./index.css";

function App() {
  const [imageUrl, setImageUrl] = useState(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageUrl(URL.createObjectURL(file));
    }
  };

  return (
    <div className="dashboard">
      <div className="sidebar">
        <h2>🧠 SAM Segmentation</h2>
        <input type="file" accept="image/*" onChange={handleUpload} />
        <p>Select an image and then brush or click to mark segmentation areas.</p>
      </div>
      <div className="main-canvas">
        {imageUrl ? (
          <CanvasSegmentor imageUrl={imageUrl} />
        ) : (
          <p className="placeholder">Upload an image to start</p>
        )}
      </div>
    </div>
  );
}

export default App;
