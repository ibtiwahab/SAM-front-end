import React, { useRef, useState, useEffect, useCallback } from "react";
import axios from "axios";

export default function CanvasSegmentor({ imageUrl }) {
  const [mode, setMode] = useState("brush"); // "hover" or "brush"
  const [brushSize, setBrushSize] = useState(15); // Default brush size increased
  const [isLoading, setIsLoading] = useState(false);
  const [resultImg, setResultImg] = useState(null);
  const [brushStrokes, setBrushStrokes] = useState([]);
  const [hoverMask, setHoverMask] = useState(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const brushCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const sliderRef = useRef(null);
  const imageRef = useRef(null);
  const isDrawing = useRef(false);
  const mousePos = useRef({ x: 0, y: 0 });
  const lastRequestRef = useRef(null);
  const hoverRequestId = useRef(0);
  const animationFrameRef = useRef(null);
  const [canvasSize, setCanvasSize] = useState({ width: 512, height: 512 });
  const [isHoverLoading, setIsHoverLoading] = useState(false);
  const mouseInCanvasRef = useRef(false);

  // Load the image
  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imageRef.current = img;
      
      // Set the canvas size based on the container and image aspect ratio
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        const imgAspectRatio = img.width / img.height;
        
        let canvasWidth = containerWidth;
        let canvasHeight = containerWidth / imgAspectRatio;
        
        // Cap the height if needed
        const maxHeight = window.innerHeight * 0.7;
        if (canvasHeight > maxHeight) {
          canvasHeight = maxHeight;
          canvasWidth = maxHeight * imgAspectRatio;
        }
        
        setCanvasSize({
          width: canvasWidth,
          height: canvasHeight
        });
      }
      
      // Upload to backend
      uploadImageToBackend(imageUrl);
    };
    
    img.onerror = () => {
      console.error("Error loading image");
    };
    
    img.src = imageUrl;
  }, [imageUrl]);

  // Initialize canvas when size changes
  useEffect(() => {
    if (!canvasRef.current || !maskCanvasRef.current || !brushCanvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const brushCanvas = brushCanvasRef.current;
    
    // Set canvas size
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    maskCanvas.width = canvasSize.width;
    maskCanvas.height = canvasSize.height;
    brushCanvas.width = canvasSize.width;
    brushCanvas.height = canvasSize.height;
    
    // Initial render
    drawCanvas();
  }, [canvasSize]);

  // Add slider event handlers
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingSlider && sliderRef.current) {
        const sliderRect = sliderRef.current.getBoundingClientRect();
        const sliderWidth = sliderRect.width;
        const x = Math.max(0, Math.min(e.clientX - sliderRect.left, sliderWidth));
        const percentage = x / sliderWidth;
        const newSize = Math.round(5 + percentage * 45); // 5 to 50
        setBrushSize(newSize);
      }
    };

    const handleMouseUp = () => {
      setIsDraggingSlider(false);
    };

    if (isDraggingSlider) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingSlider]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (lastRequestRef.current) {
        try {
          lastRequestRef.current.abort();
        } catch (error) {
          console.log('Error aborting request:', error);
        }
      }
    };
  }, []);

  // Draw the main canvas with image
  const drawCanvas = useCallback(() => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw the image
    ctx.drawImage(
      imageRef.current, 
      0, 0, 
      canvas.width, 
      canvas.height
    );
  }, []);

  // Update brush canvas
  const updateBrushCanvas = useCallback(() => {
    if (!brushCanvasRef.current) return;
    
    const ctx = brushCanvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, brushCanvasRef.current.width, brushCanvasRef.current.height);
    
    // Draw all brush strokes
    brushStrokes.forEach(stroke => {
      if (stroke.points.length < 2) return;
      
      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      
      for (let i = 1; i < stroke.points.length; i++) {
        ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
      
      ctx.strokeStyle = '#ff3366'; // Modern pink
      ctx.lineWidth = stroke.size;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.stroke();
    });
  }, [brushStrokes]);

  // Update mask canvas with hover mask
  const updateMaskCanvas = useCallback(() => {
    if (!maskCanvasRef.current) return;
    
    const ctx = maskCanvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
    
    if (hoverMask) {
      ctx.drawImage(hoverMask, 0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
    }
  }, [hoverMask]);

  // Effect to update canvases
  useEffect(() => {
    drawCanvas();
    updateBrushCanvas();
    updateMaskCanvas();
  }, [drawCanvas, updateBrushCanvas, updateMaskCanvas, brushStrokes, hoverMask]);

  // Upload image to backend
  const uploadImageToBackend = async (imageUrl) => {
    try {
      setIsLoading(true);
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append('file', blob);
      
      await axios.post('http://localhost:8000/upload', formData);
      console.log('Image uploaded successfully');
      setIsLoading(false);
    } catch (error) {
      console.error('Error uploading image:', error);
      setIsLoading(false);
    }
  };

  // Get alpha mask from the hovered point
  const getAlphaMask = useCallback(async (point) => {
    if (!imageRef.current || !point) return;
    
    setIsLoading(true);
    
    try {
      // Calculate scaled coordinates
      const scaleX = imageRef.current.width / canvasSize.width;
      const scaleY = imageRef.current.height / canvasSize.height;
      
      const scaledPoint = {
        x: Math.round(point.x * scaleX),
        y: Math.round(point.y * scaleY)
      };
      
      // Request the alpha mask from the backend
      const response = await axios.post(
        'http://localhost:8000/alpha_mask',
        { point: scaledPoint },
        { responseType: 'blob' }
      );
      
      // Set the result image
      const url = URL.createObjectURL(response.data);
      setResultImg(url);
    } catch (error) {
      console.error('Error getting alpha mask:', error);
      alert('Failed to get mask. See console for details.');
    } finally {
      setIsLoading(false);
    }
  }, [canvasSize]);

  // Request hover preview - optimized with AbortController for cancelable requests
  const requestHoverPreview = useCallback(async (point) => {
    if (!imageRef.current || !point || !mouseInCanvasRef.current) return;
    
    // Prevent too frequent updates which cause flickering
    setIsHoverLoading(true);
    
    // Cancel any pending request
    if (lastRequestRef.current) {
      try {
        lastRequestRef.current.abort();
      } catch (error) {
        console.log('Error aborting request:', error);
      }
    }
    
    // Create a new AbortController for this request
    const controller = new AbortController();
    lastRequestRef.current = controller;
    
    // Generate a unique ID for this request to track it
    const requestId = ++hoverRequestId.current;
    
    try {
      // Calculate scaled coordinates
      const scaleX = imageRef.current.width / canvasSize.width;
      const scaleY = imageRef.current.height / canvasSize.height;
      
      const scaledPoint = {
        x: Math.round(point.x * scaleX),
        y: Math.round(point.y * scaleY)
      };
      
      // Make the request with signal for abortion
      const response = await axios.post(
        'http://localhost:8000/hover_preview',
        { point: scaledPoint },
        {
          responseType: 'blob',
          signal: controller.signal
        }
      );
      
      // If this isn't the latest request or mouse is no longer in canvas, ignore the result
      if (requestId !== hoverRequestId.current || !mouseInCanvasRef.current) return;
      
      // Create an image from the blob
      const maskBlob = response.data;
      const maskUrl = URL.createObjectURL(maskBlob);
      
      const maskImg = new Image();
      maskImg.onload = () => {
        // If this isn't the latest request anymore, clean up and return
        if (requestId !== hoverRequestId.current || !mouseInCanvasRef.current) {
          URL.revokeObjectURL(maskUrl);
          return;
        }
        
        setHoverMask(maskImg);
        URL.revokeObjectURL(maskUrl);
        setIsHoverLoading(false);
      };
      
      maskImg.onerror = () => {
        console.error('Error loading mask image');
        setIsHoverLoading(false);
        URL.revokeObjectURL(maskUrl);
      };
      
      maskImg.src = maskUrl;
    } catch (error) {
      // Ignore aborted request errors
      if (error.name === 'AbortError' || error.name === 'CanceledError') {
        console.log('Hover request aborted');
        return;
      }
      console.error('Error getting hover preview:', error);
      setIsHoverLoading(false);
    }
  }, [canvasSize]);

  // Handle mouse enter
  const handleMouseEnter = useCallback(() => {
    mouseInCanvasRef.current = true;
    
    // Reset hover loading state
    setIsHoverLoading(false);
  }, []);

  // Handle mouse movements
  const handleMouseMove = useCallback((e) => {
    if (!canvasRef.current) return;
    
    // Ensure mouse is marked as in canvas
    mouseInCanvasRef.current = true;
    
    // Get the mouse position relative to the canvas
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    mousePos.current = { x, y };
    
    if (mode === "hover") {
      // Use requestAnimationFrame to throttle hover requests 
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      animationFrameRef.current = requestAnimationFrame(() => {
        if (!isHoverLoading) {
          requestHoverPreview(mousePos.current);
        }
      });
    } else if (mode === "brush" && isDrawing.current) {
      // Add point to current brush stroke
      const newStrokes = [...brushStrokes];
      const currentStroke = newStrokes[newStrokes.length - 1];
      currentStroke.points.push({ x, y });
      setBrushStrokes(newStrokes);
    }
  }, [mode, isHoverLoading, requestHoverPreview, brushStrokes]);

  // Handle mouse down for drawing
  const handleMouseDown = useCallback((e) => {
    if (!canvasRef.current || isLoading) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    if (mode === "hover") {
      // When in hover mode and clicking, get the alpha mask for the current position
      if (hoverMask && mouseInCanvasRef.current) {
        getAlphaMask(mousePos.current);
      }
    } else if (mode === "brush") {
      isDrawing.current = true;
      // Start a new brush stroke
      setBrushStrokes([...brushStrokes, { 
        points: [{ x, y }],
        size: brushSize
      }]);
    }
  }, [mode, isLoading, hoverMask, brushStrokes, brushSize, getAlphaMask]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    isDrawing.current = false;
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    // Mark mouse as outside canvas
    mouseInCanvasRef.current = false;
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    // Cancel any in-flight request
    if (lastRequestRef.current) {
      try {
        lastRequestRef.current.abort();
      } catch (error) {
        console.log('Error aborting request:', error);
      }
    }
    
    // Clear hover mask
    setHoverMask(null);
    setIsHoverLoading(false);
    isDrawing.current = false;
  }, []);

  // Generate alpha mask directly from brush strokes
  const generateBrushAlphaMask = useCallback(() => {
    if (brushStrokes.length === 0) {
      alert('Please draw with the brush before getting the mask');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Create a temporary canvas for the mask
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvasSize.width;
      tempCanvas.height = canvasSize.height;
      const ctx = tempCanvas.getContext('2d');
      
      // Clear the canvas with a transparent background
      ctx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
      
      // Set up for alpha mask creation
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
      
      // Draw all brush strokes in white
      ctx.globalCompositeOperation = 'source-over';
      brushStrokes.forEach(stroke => {
        if (stroke.points.length < 2) return;
        
        ctx.beginPath();
        ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
        
        for (let i = 1; i < stroke.points.length; i++) {
          ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
        
        ctx.strokeStyle = '#ffffff'; // White for the mask
        ctx.lineWidth = stroke.size;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
      });
      
      // Convert the canvas to an image
      const maskDataUrl = tempCanvas.toDataURL('image/png');
      setResultImg(maskDataUrl);
      
    } catch (error) {
      console.error('Error generating brush mask:', error);
      alert('Failed to generate mask. See console for details.');
    } finally {
      setIsLoading(false);
    }
  }, [brushStrokes, canvasSize]);

  // Segment the image based on mode
  const handleSegment = async () => {
    if (mode === "brush") {
      // For brush mode, generate alpha mask directly from brush strokes
      generateBrushAlphaMask();
    } else if (mode === "hover") {
      // In hover mode, get the alpha mask for the current hover point
      if (mousePos.current && mouseInCanvasRef.current) {
        getAlphaMask(mousePos.current);
      }
    }
  };

  // Clear all brush strokes and results
  const clearCanvas = useCallback(() => {
    setBrushStrokes([]);
    setResultImg(null);
    setHoverMask(null);
    setIsHoverLoading(false);
    
    // Redraw the canvas
    drawCanvas();
  }, [drawCanvas]);

  // Brush cursor indicator
  const BrushCursor = () => {
    if (mode !== "brush" || !mouseInCanvasRef.current) return null;
    
    return (
      <div 
        className="absolute rounded-full pointer-events-none z-10 border-2 border-dashed border-pink-500 bg-transparent"
        style={{ 
          left: mousePos.current.x,
          top: mousePos.current.y,
          width: `${brushSize}px`,
          height: `${brushSize}px`,
          transform: 'translate(-50%, -50%)'
        }}
      />
    );
  };

  // Hover cursor indicator
  const HoverCursor = () => {
    if (mode !== "hover" || !mouseInCanvasRef.current) return null;
    
    return (
      <div 
        className="absolute w-1 h-1 -ml-0.5 -mt-0.5 pointer-events-none z-10"
        style={{ 
          left: mousePos.current.x,
          top: mousePos.current.y,
          display: isHoverLoading ? 'none' : 'block'
        }}
      >
        <div className="w-full h-full rounded-full border border-yellow-500 bg-yellow-400"></div>
      </div>
    );
  };

  // Calculate current slider position
  const sliderPosition = ((brushSize - 5) / 45) * 100; // Convert 5-50 to 0-100%

  return (
    <div className="w-full h-full flex flex-col">
      {/* Toolbar */}
      <div className="mb-4 p-3 bg-gray-800 rounded-md shadow-lg flex flex-wrap items-center">
        <button
          className={`mr-2 px-4 py-2 rounded-md ${mode === "hover" ? "bg-indigo-500 text-white" : "bg-gray-600 text-white hover:bg-gray-700"} transition-colors`}
          onClick={() => setMode("hover")}
        >
          👆 Hover Mode
        </button>
        
        <button
          className={`mr-2 px-4 py-2 rounded-md ${mode === "brush" ? "bg-indigo-500 text-white" : "bg-gray-600 text-white hover:bg-gray-700"} transition-colors`}
          onClick={() => setMode("brush")}
        >
          🖌️ Brush Mode
        </button>
        
        {mode === "brush" && (
          <div className="flex items-center mr-4 px-2 py-1 bg-gray-700 rounded-md">
            <span className="text-white text-sm mr-2">Size: {brushSize}px</span>
            <div 
              ref={sliderRef}
              className="w-32 h-2 bg-gray-600 rounded-full relative cursor-pointer"
              onMouseDown={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const percentage = x / rect.width;
                const newSize = Math.round(5 + percentage * 45); // 5 to 50
                setBrushSize(newSize);
                setIsDraggingSlider(true);
              }}
            >
              <div 
                className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 to-pink-500 rounded-full"
                style={{ width: `${sliderPosition}%` }}
              ></div>
              <div
                className="absolute top-0 w-4 h-4 bg-white rounded-full shadow-md -mt-1 cursor-grab"
                style={{ 
                  left: `calc(${sliderPosition}% - 8px)`,
                  cursor: isDraggingSlider ? 'grabbing' : 'grab'
                }}
              ></div>
            </div>
          </div>
        )}
        
        <button 
          onClick={clearCanvas}
          className="mr-2 px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors"
        >
          Clear
        </button>
        
        <button 
          onClick={handleSegment}
          className="mr-2 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? "Processing..." : "Get Mask"}
        </button>
        
        {mode === "brush" && (
          <div className="ml-2 text-sm text-white">
            <b>Instructions:</b> Draw over the areas you want to mask, then click "Get Mask"
          </div>
        )}
        
        {mode === "hover" && (
          <div className="ml-2 text-sm text-white">
            <b>Instructions:</b> Hover over objects to preview, click to select
          </div>
        )}
      </div>

      {/* Canvas Section */}
      <div 
        ref={containerRef} 
        className="relative mb-4 flex justify-center"
      >
        <div className="relative">
          {/* Main Canvas for Image */}
          <canvas
            ref={canvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className="border border-gray-300 rounded-md bg-gray-50"
            onMouseEnter={handleMouseEnter}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
          />
          
          {/* Brush Canvas (for brush strokes) */}
          <canvas
            ref={brushCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className="absolute top-0 left-0 pointer-events-none"
          />
          
          {/* Mask Canvas (for hover preview) */}
          <canvas
            ref={maskCanvasRef}
            width={canvasSize.width}
            height={canvasSize.height}
            className="absolute top-0 left-0 pointer-events-none"
          />
          
          {/* Brush Cursor */}
          <BrushCursor />
          
          {/* Hover Cursor */}
          <HoverCursor />
          
          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-md backdrop-filter backdrop-blur-sm">
              <div className="bg-white p-4 rounded-lg shadow-lg flex items-center">
                <div className="animate-spin w-6 h-6 border-4 border-indigo-500 border-t-transparent rounded-full mr-3"></div>
                <span className="text-gray-800 font-medium">Processing...</span>
              </div>
            </div>
          )}
          
          {/* Small Hover Loading Indicator */}
          {isHoverLoading && mode === "hover" && mouseInCanvasRef.current && (
            <div 
              className="absolute pointer-events-none"
              style={{ 
                left: `${mousePos.current.x}px`, 
                top: `${mousePos.current.y}px`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          )}
        </div>
      </div>

      {/* Result Section with Black Background */}
      {resultImg && (
        <div className="w-full border border-gray-700 rounded-md p-4 bg-black shadow-lg">
          <h3 className="text-lg font-semibold mb-2 text-center text-white">Brush Mask Result</h3>
          <div className="flex justify-center">
            <div className="relative">
              <img 
                src={resultImg} 
                alt="Brush Mask Result" 
                className="max-w-full max-h-80 object-contain"
              />
            </div>
          </div>
          <p className="text-center mt-2 text-sm text-gray-400">
            White areas represent the brush strokes mask.
          </p>
        </div>
      )}
    </div>
  );
}