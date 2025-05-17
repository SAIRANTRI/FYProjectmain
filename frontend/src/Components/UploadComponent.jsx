import { useEffect, useState, useCallback } from "react";
import { Upload, X, AlertCircle, Check } from "react-feather";
import { useUploadStore } from "../store/useUploadStore";
import { useUserStore } from "../store/useUserStore";
import Toast from "./Toast";

export default function UploadComponent() {
  const {
    referenceImages,
    poolImages,
    uploadReferenceImages,
    uploadPoolImages,
    fetchUploadedImages,
    deleteImage,
    loading,
    uploadProgress,
  } = useUploadStore();

  const [isDraggingRef, setIsDraggingRef] = useState(false);
  const [isDraggingPool, setIsDraggingPool] = useState(false);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationError, setClassificationError] = useState(null);
  const [toast, setToast] = useState({ show: false, message: "", type: "success" });

  const { fetchProfile } = useUserStore();

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  useEffect(() => {
    fetchUploadedImages();
  }, [fetchUploadedImages]);

  const showToast = (message, type = "success") => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: "", type: "success" }), 3000);
  };

  const handleUpload = async (event, type) => {
    const files = Array.from(event.target.files);
    
    // Validate files
    const validFiles = files.filter(file => {
      const isImage = file.type.startsWith('image/');
      const isValidSize = file.size <= 5 * 1024 * 1024; // 5MB limit
      
      if (!isImage) {
        showToast(`${file.name} is not an image file`, "error");
      } else if (!isValidSize) {
        showToast(`${file.name} exceeds 5MB size limit`, "error");
      }
      
      return isImage && isValidSize;
    });
    
    if (validFiles.length === 0) return;
    
    try {
      if (type === "reference") {
        // For reference images, only allow one at a time
        if (referenceImages.length > 0) {
          showToast("Please delete the existing reference image first", "error");
          return;
        }
        await uploadReferenceImages(validFiles);
        showToast("Reference image uploaded successfully");
      } else {
        await uploadPoolImages(validFiles);
        showToast("Pool images uploaded successfully");
      }
    } catch (error) {
      showToast(`Upload failed: ${error.message}`, "error");
    }
  };

  const handleDragOver = useCallback((e, type) => {
    e.preventDefault();
    if (type === "reference") {
      setIsDraggingRef(true);
    } else {
      setIsDraggingPool(true);
    }
  }, []);

  const handleDragLeave = useCallback((e, type) => {
    e.preventDefault();
    if (type === "reference") {
      setIsDraggingRef(false);
    } else {
      setIsDraggingPool(false);
    }
  }, []);

  const handleDrop = useCallback(async (e, type) => {
    e.preventDefault();
    if (type === "reference") {
      setIsDraggingRef(false);
      
      // For reference images, only allow one at a time
      if (referenceImages.length > 0) {
        showToast("Please delete the existing reference image first", "error");
        return;
      }
    } else {
      setIsDraggingPool(false);
    }
    
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    
    if (files.length === 0) {
      showToast("Please drop image files only", "error");
      return;
    }
    
    try {
      if (type === "reference") {
        await uploadReferenceImages(files);
        showToast("Reference image uploaded successfully");
      } else {
        await uploadPoolImages(files);
        showToast("Pool images uploaded successfully");
      }
    } catch (error) {
      showToast(`Upload failed: ${error.message}`, "error");
    }
  }, [uploadReferenceImages, uploadPoolImages, referenceImages.length]);

  const handleDelete = async (type, imageId) => {
    try {
      await deleteImage(imageId, type);
      showToast(`Image deleted successfully`);
    } catch (error) {
      showToast(`Delete failed: ${error.message}`, "error");
    }
  };

  const handleDownload = async () => {
    if (referenceImages.length === 0) {
      setClassificationError("Please upload a reference image");
      return;
    }

    if (poolImages.length === 0) {
      setClassificationError("Please upload at least one pool image");
      return;
    }

    try {
      setIsClassifying(true);
      setClassificationError(null);

      // Create FormData object
      const formData = new FormData();
      
      // Get the reference image
      const referenceImage = referenceImages[0];
      
      // Fetch the reference image file
      const referenceResponse = await fetch(referenceImage.imageUrl);
      const referenceBlob = await referenceResponse.blob();
      formData.append('reference_image', new File([referenceBlob], 'reference.jpg', { type: 'image/jpeg' }));
      
      // Fetch and append all pool images
      for (let i = 0; i < poolImages.length; i++) {
        const poolImage = poolImages[i];
        const poolResponse = await fetch(poolImage.imageUrl);
        const poolBlob = await poolResponse.blob();
        formData.append('pool_images', new File([poolBlob], `pool_${i}.jpg`, { type: 'image/jpeg' }));
      }
      
      // Add user ID
      formData.append('user_id', localStorage.getItem('userId') || 'anonymous');
      
      // Send request to FastAPI backend
      const response = await fetch('http://localhost:8000/api/classify/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Classification failed');
      }
      
      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'classified_images.zip';
      document.body.appendChild(a);
      a.click();
      
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      setIsClassifying(false);
      showToast("Classification completed successfully");
    } catch (error) {
      console.error('Error during classification:', error);
      setClassificationError(error.message || 'Failed to classify images');
      setIsClassifying(false);
      showToast(`Classification failed: ${error.message}`, "error");
    }
  };

  return (
    <div className="min-h-screen text-white flex flex-col pb-28 items-center">
      {toast.show && (
        <Toast 
          message={toast.message} 
          type={toast.type} 
          onClose={() => setToast({ show: false, message: "", type: "success" })} 
        />
      )}
      
      <div className="w-full max-w-[1200px] p-5">
        <h1 className="text-3xl font-extrabold mb-6 text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
          Image Classification
        </h1>
        
        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Left Column (span 4) */}
          <div className="md:col-span-4 flex flex-col space-y-6">
            {/* Reference Image Section */}
            <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700 hover:border-purple-500 transition-all duration-300 h-[350px] flex flex-col">
              <h2 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
                Reference Image
              </h2>
              
              <div className="flex-grow flex flex-col justify-center">
                {referenceImages.length === 0 ? (
                  <div
                    onDragOver={(e) => handleDragOver(e, "reference")}
                    onDragLeave={(e) => handleDragLeave(e, "reference")}
                    onDrop={(e) => handleDrop(e, "reference")}
                    className={`w-full h-full border-2 border-dashed p-8 rounded-xl transition-all duration-300 text-center cursor-pointer flex flex-col items-center justify-center ${
                      isDraggingRef
                        ? "scale-105 border-purple-500 bg-gray-800/30"
                        : "hover:bg-gray-800/20 hover:border-purple-400"
                    }`}
                  >
                    <Upload className="w-10 h-10 mb-3 text-purple-500" />
                    <p className="text-sm mb-2">Drag and drop your reference image here</p>
                    <p className="text-xs text-gray-400">or</p>
                    <button
                      onClick={() => document.getElementById("referenceUpload").click()}
                      className="mt-3 px-4 py-1.5 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white rounded-md shadow-md text-xs transform hover:translate-y-[-2px] active:translate-y-[1px] transition-all duration-300"
                    >
                      Browse Files
                    </button>
                    <input
                      type="file"
                      onChange={(e) => handleUpload(e, "reference")}
                      id="referenceUpload"
                      className="hidden"
                      accept="image/*"
                    />
                    <p className="mt-3 text-xs text-gray-400">Supported formats: JPG, PNG, GIF (max 5MB)</p>
                  </div>
                ) : (
                  <div className="relative bg-gray-800 rounded-lg overflow-hidden h-full group flex items-center justify-center">
                    <img
                      src={referenceImages[0].imageUrl}
                      alt="reference"
                      className="max-w-full max-h-full object-contain"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center">
                      <button
                        onClick={() => handleDelete("reference", referenceImages[0]._id)}
                        className="opacity-0 group-hover:opacity-100 bg-red-600 text-white rounded-full p-2 hover:bg-red-700 transition-all duration-300 transform hover:scale-110"
                      >
                        <X size={18} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
              
              {uploadProgress > 0 && uploadProgress < 100 && (
                <div className="w-full mt-4">
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">Uploading: {uploadProgress}%</p>
                </div>
              )}
            </div>
            
            {/* Pool Images Upload Button */}
            <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700 hover:border-purple-500 transition-all duration-300 h-[350px] flex flex-col">
              <h2 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
                Upload Pool Images
              </h2>
              
              <div
                onDragOver={(e) => handleDragOver(e, "pool")}
                onDragLeave={(e) => handleDragLeave(e, "pool")}
                onDrop={(e) => handleDrop(e, "pool")}
                className={`w-full flex-grow border-2 border-dashed rounded-xl transition-all duration-300 text-center cursor-pointer flex flex-col items-center justify-center ${
                  isDraggingPool
                    ? "scale-105 border-purple-500 bg-gray-800/30"
                    : "hover:bg-gray-800/20 hover:border-purple-400"
                }`}
              >
                <Upload className="w-10 h-10 mb-3 text-purple-500" />
                <p className="text-sm mb-2">Drag and drop your pool images here</p>
                <p className="text-xs text-gray-400">or</p>
                <button
                  onClick={() => document.getElementById("poolUpload").click()}
                  className="mt-3 px-4 py-1.5 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white rounded-md shadow-md text-xs transform hover:translate-y-[-2px] active:translate-y-[1px] transition-all duration-300"
                >
                  Browse Files
                </button>
                <input
                  type="file"
                  multiple
                  onChange={(e) => handleUpload(e, "pool")}
                  id="poolUpload"
                  className="hidden"
                  accept="image/*"
                />
                <p className="mt-3 text-xs text-gray-400">Supported formats: JPG, PNG, GIF (max 5MB)</p>
              </div>
            </div>
          </div>
          
          {/* Right Column (span 8) */}
          <div className="md:col-span-8 flex flex-col space-y-6">
            {/* Pool Images Display */}
            <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700 hover:border-purple-500 transition-all duration-300 h-[350px] flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
                  Pool Images ({poolImages.length})
                </h2>
                
                {poolImages.length > 0 && (
                  <span className="text-xs text-gray-400">
                    {poolImages.length} image{poolImages.length !== 1 ? 's' : ''} uploaded
                  </span>
                )}
              </div>
              
              <div className="flex-grow overflow-auto">
                {poolImages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full border-2 border-dashed border-gray-700 rounded-xl">
                    <p className="text-gray-400 text-sm">No pool images uploaded yet</p>
                    <p className="text-gray-500 text-xs mt-2">Upload images using the panel on the left</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 h-full overflow-auto p-1">
                    {poolImages.map((img) => (
                      <div
                        key={img._id}
                        className="relative bg-gray-800 rounded-lg overflow-hidden h-24 group transform transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-purple-500/20"
                      >
                        <img
                          src={img.imageUrl}
                          alt="pool"
                          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center">
                          <button
                            onClick={() => handleDelete("pool", img._id)}
                            className="opacity-0 group-hover:opacity-100 bg-red-600 text-white rounded-full p-1.5 hover:bg-red-700 transition-all duration-300 transform hover:scale-110"
                          >
                            <X size={16} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Results Section */}
            <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700 hover:border-purple-500 transition-all duration-300 h-[350px] flex flex-col">
              <h2 className="text-xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
                Results
              </h2>
              
              <div className="bg-gray-800/50 p-4 rounded-lg text-center flex-grow flex flex-col justify-center">
                {classificationError ? (
                  <div className="flex items-center justify-center text-red-400">
                    <AlertCircle size={20} className="mr-2" />
                    <p>{classificationError}</p>
                  </div>
                ) : (
                  <p className="text-sm">
                    {isClassifying 
                      ? "Processing images... Please wait." 
                      : "Click the Download button below to classify and download results."}
                  </p>
                )}
                
                {referenceImages.length > 0 && poolImages.length > 0 && !isClassifying && !classificationError && (
                  <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-md inline-flex items-center mx-auto">
                    <Check size={18} className="text-green-500 mr-2" />
                    <span className="text-green-400">Ready to classify {poolImages.length} images</span>
                  </div>
                )}
                
                {/* Progress Bar */}
                {(loading || isClassifying) && (
                  <div className="w-full mt-4">
                    <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-pink-500 to-purple-500 rounded-full animate-pulse"
                        style={{ width: "100%" }}
                      />
                    </div>
                    <p className="text-center text-xs text-gray-400 mt-2">
                      {isClassifying ? "Classifying images..." : "Processing..."}
                    </p>
                  </div>
                )}
                
                {/* Download Button */}
                <div className="mt-6 flex justify-center">
                  <button
                    onClick={handleDownload}
                    disabled={loading || isClassifying || referenceImages.length === 0 || poolImages.length === 0}
                    className={`flex items-center justify-center px-6 py-3 rounded-lg bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-base cursor-pointer transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] transform hover:translate-y-[-2px] active:translate-y-[1px] ${
                      (loading || isClassifying || referenceImages.length === 0 || poolImages.length === 0) 
                        ? "opacity-50 cursor-not-allowed" 
                        : ""
                    }`}
                  >
                    <span className="text-gray-200 mr-2">
                      {isClassifying ? "Processing..." : "Download Results"}
                    </span>
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-white/10 shadow-inner">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="w-5 h-5 text-pink-400"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m0 0l-4-4m4 4l4-4M4 20h16" />
                      </svg>
                    </span>
                  </button>
                </div>
                
                <p className="mt-2 text-xs text-center text-gray-400">
                  {referenceImages.length === 0 
                    ? "Upload a reference image to continue" 
                    : poolImages.length === 0 
                      ? "Upload pool images to continue" 
                      : "Click to process and download results"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}