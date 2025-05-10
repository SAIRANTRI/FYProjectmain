import { useState, useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faFolder } from "@fortawesome/free-solid-svg-icons";
import { faGoogleDrive } from "@fortawesome/free-brands-svg-icons";
import GooglePhotos from "../assets/googlephotos.svg";
import downloadIcon from "../assets/download.svg";
import { Upload } from "react-feather";


export default function UploadComponent() {
  const [images, setImages] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const savedImages = localStorage.getItem("uploadedImages");
    if (savedImages) setImages(JSON.parse(savedImages));
  }, []);

  useEffect(() => {
    if (images.length > 0) {
      localStorage.setItem("uploadedImages", JSON.stringify(images));
    }
  }, [images]);

  const handleImageUpload = async (event) => {
    const files = Array.from(event.target.files);
    const base64Promises = files.map((file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    });

    const base64Images = await Promise.all(base64Promises);
    setImages((prev) => [...prev, ...base64Images]);
    simulateUploadProgress();
  };

  const simulateUploadProgress = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setUploadProgress(progress);
      if (progress >= 100) {
        clearInterval(interval);
        setTimeout(() => setUploadProgress(0), 1000);
      }
    }, 150);
  };

  const handleDownload = () => alert("Download functionality here.");
  const handleLocalUpload = () => document.getElementById("imageUpload").click();
  const handleGoogleDriveUpload = () => alert("Google Drive upload clicked!");
  const handlePhotosUpload = () => alert("Google Photos upload clicked!");

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files).filter((file) =>
      file.type.startsWith("image/")
    );

    const base64Promises = droppedFiles.map((file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    });

    const base64Images = await Promise.all(base64Promises);
    setImages((prev) => [...prev, ...base64Images]);
    simulateUploadProgress();
  };

  return (
    <div className="min-h-screen text-white flex flex-col pb-28 items-center">
      <div className="flex-grow flex flex-col p-5 w-full max-w-[1048px] items-center space-y-6">
        {/* Upload Options */}
        <div className="flex flex-wrap justify-center gap-12 mb-6">
          {[{ icon: GooglePhotos, label: "Google Photos", action: handlePhotosUpload },
            { icon: faFolder, label: "Local Upload", action: handleLocalUpload },
            { icon: faGoogleDrive, label: "Google Drive", action: handleGoogleDriveUpload }].map((item, idx) => (
            <button key={idx} onClick={item.action} className="flex flex-col items-center transition hover:scale-105">
              {typeof item.icon === "string" ? (
                <img src={item.icon} alt={item.label} className="w-[40px] h-[40px] filter invert" />
              ) : (
                <FontAwesomeIcon icon={item.icon} className="text-white text-4xl" />
              )}
              <span className="mt-2 text-sm">{item.label}</span>
            </button>
          ))}
        </div>

        <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
          Upload Images
        </h1>

        {/* Drag and Drop */}
        <div
          className={`w-full max-w-[1048px] border-2 border-dashed border-gray-500 p-12 rounded-xl flex flex-col items-center justify-center transition-all duration-300 cursor-pointer ${
            isDragging ? "scale-105 border-purple-500 bg-gray-800/10" : "hover:bg-gray-800/10"
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className="w-12 h-12 mb-4 text-purple-500" />
          <p className="text-lg mb-2">Drag and drop your images here</p>
          <p className="text-sm text-gray-400">or</p>
          <button
            onClick={handleLocalUpload}
            className="mt-4 px-6 py-2 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] text-sm font-medium"
          >
            Browse Files
          </button>
          <input type="file" multiple onChange={handleImageUpload} className="hidden" id="imageUpload" accept="image/*" />
        </div>

        {/* Gallery */}
        {images.length > 0 && (
          <div className="w-full max-w-[1048px] mt-10">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              {images.map((image, index) => (
                <div key={index} className="overflow-hidden bg-gray-800 h-32 rounded-lg shadow-md">
                  <img src={image} alt={`Uploaded ${index + 1}`} className="h-full w-full object-cover rounded-lg transition-transform duration-300 hover:scale-105" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Progress Bar */}
        {uploadProgress > 0 && (
          <div className="w-full max-w-[1048px] mt-8">
            <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden shadow-inner">
              <div
                className="h-full bg-gradient-to-r from-pink-500 to-purple-500 rounded-full animate-pulse"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Zip Section */}
        <div className="w-full max-w-[1048px] mt-14">
          <div className="shadow-[0px_0px_27px_0px_rgba(173,40,40,0.48)] rounded-2xl bg-[linear-gradient(rgba(22,4,8,0.5),rgba(22,4,8,0.5))] p-[8px]">
            <div className="rounded-xl border border-white/10 bg-gray-900/90 flex flex-col sm:flex-row justify-between items-center p-4 space-y-4 sm:space-y-0 sm:space-x-6">
              <div className="flex flex-col">
                <div className="text-lg font-semibold">Zip File Name</div>
                <span className="text-sm text-pink-100/80">Size — {images.length} images</span>
              </div>
              <div
                onClick={handleDownload}
                className="flex items-center px-4 py-2 rounded bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-sm cursor-pointer transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)]"
              >
                <span className="text-gray-200 mr-2">Download</span>
                <img src={downloadIcon} className="w-[9.2px] h-[5.7px] rotate-[-90deg]" alt="download" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
