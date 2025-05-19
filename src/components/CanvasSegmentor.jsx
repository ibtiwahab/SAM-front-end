import React, { useRef, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line, Circle } from "react-konva";
import useImage from "use-image";
import axios from "axios";

export default function CanvasSegmentor({ imageUrl }) {
  const [mode, setMode] = useState("brush"); // "brush" or "point"
  const [lines, setLines] = useState([]);
  const [points, setPoints] = useState([]);
  const isDrawing = useRef(false);
  const [img, status] = useImage(imageUrl);
  const stageRef = useRef();

  const handleMouseDown = () => {
    if (mode === "brush") {
      isDrawing.current = true;
      const pos = stageRef.current.getPointerPosition();
      setLines([...lines, { points: [pos.x, pos.y] }]);
    }
  };

  const handleMouseMove = () => {
    if (!isDrawing.current || mode !== "brush") return;
    const stage = stageRef.current;
    const point = stage.getPointerPosition();
    const lastLine = lines[lines.length - 1];
    lastLine.points = lastLine.points.concat([point.x, point.y]);
    lines.splice(lines.length - 1, 1, lastLine);
    setLines(lines.concat());
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  const handleClick = () => {
    if (mode === "point") {
      const pos = stageRef.current.getPointerPosition();
      setPoints([...points, pos]);
    }
  };

  const clearCanvas = () => {
    setLines([]);
    setPoints([]);
  };

  const sendToBackend = async () => {
    const uri = stageRef.current.toDataURL({ mimeType: "image/png" });
    const blob = await fetch(uri).then(res => res.blob());
    const form = new FormData();
    form.append("brushed_mask", blob, "mask.png");

    const res = await axios.post("http://localhost:8000/segment", form, {
      responseType: "blob",
    });
    const url = URL.createObjectURL(res.data);
    window.open(url, "_blank");
  };

  if (status === "loading") return <p>Loading image...</p>;
  if (status === "failed") return <p>Failed to load image</p>;

  return (
    <div className="canvas-tools">
      <div className="toolbar">
        <button
          className={mode === "brush" ? "active" : ""}
          onClick={() => setMode("brush")}
        >
          🖌️ Brush Mode
        </button>
        <button
          className={mode === "point" ? "active" : ""}
          onClick={() => setMode("point")}
        >
          📍 Point Mode
        </button>
        <button onClick={clearCanvas}>🧼 Clear</button>
        <button onClick={sendToBackend}>📤 Segment</button>
      </div>

      <Stage
        width={512}
        height={512}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        ref={stageRef}
        style={{ border: "1px solid #ccc", background: "#f8f8f8" }}
      >
        <Layer>
          {img && <KonvaImage image={img} />}
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line.points}
              stroke="red"
              strokeWidth={5}
              tension={0.5}
              lineCap="round"
              globalCompositeOperation="source-over"
            />
          ))}
          {points.map((pt, i) => (
            <Circle key={i} x={pt.x} y={pt.y} radius={6} fill="blue" />
          ))}
        </Layer>
      </Stage>
    </div>
  );
}
