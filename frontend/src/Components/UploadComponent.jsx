import { useEffect, useState } from "react";
import { Upload, X } from "react-feather";
import { useUploadStore } from "../store/useUploadStore";
import downloadIcon from "../assets/download.svg";

export default function UploadComponent() {
  const {
    referenceImages,
    poolImages,
    uploadReferenceImages,
    uploadPoolImages,
    fetchUploadedImages,
    deleteImage, // <-- make sure this exists in your store
    loading,
    error,
  } = useUploadStore();

  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    fetchUploadedImages();
  }, [fetchUploadedImages]);

  const handleUpload = async (event, type) => {
    const files = Array.from(event.target.files);
    if (type === "reference") {
      await uploadReferenceImages(files);
    } else {
      await uploadPoolImages(files);
    }
  };

  const handleDrop = async (e, type) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    if (type === "reference") {
      await uploadReferenceImages(files);
    } else {
      await uploadPoolImages(files);
    }
  };

  const handleDelete = async (type, imageId) => {
    await deleteImage(imageId, type);
  };

  const handleDownload = () => {
    console.log("Download results as ZIP");
  };
  return (
    <div className="min-h-screen text-white flex flex-col pb-28 items-center">
      <div className="w-full max-w-[1048px] p-5 flex flex-col items-center space-y-6">
        {/* Upload Reference Image */}
        <div className="w-full text-center">
          <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
            Upload Reference Image
          </h1>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setIsDragging(false);
            }}
            onDrop={(e) => handleDrop(e, "reference")}
            className={`w-full border-2 border-dashed p-12 rounded-xl transition-all duration-300 text-center cursor-pointer ${
              isDragging
                ? "scale-105 border-purple-500 bg-gray-800/10"
                : "hover:bg-gray-800/10"
            }`}
          >
            <Upload className="w-12 h-12 mb-4 text-purple-500 mx-auto" />
            <p className="text-lg mb-2">Drag and drop your reference images here</p>
            <p className="text-sm text-gray-400">or</p>
            <button
              onClick={() => document.getElementById("referenceUpload").click()}
              className="mt-4 px-6 py-2 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] text-white rounded-md shadow-md text-sm"
            >
              Browse Files
            </button>
            <input
              type="file"
              multiple
              onChange={(e) => handleUpload(e, "reference")}
              id="referenceUpload"
              className="hidden"
              accept="image/*"
            />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
            {referenceImages.map((img) => (
              <div
                key={img._id}
                className="relative bg-gray-800 rounded-lg overflow-hidden h-32"
              >
                <img
                  src={img.imageUrl}
                  alt="reference"
                  className="w-full h-full object-cover hover:scale-105 transition"
                />
                <button
                  onClick={() => handleDelete("reference", img._id)}
                  className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 hover:bg-red-700"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Upload Pool Images */}
        <div className="w-full text-center">
          <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
            Upload Pool Images
          </h1>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setIsDragging(false);
            }}
            onDrop={(e) => handleDrop(e, "pool")}
            className={`w-full border-2 border-dashed p-12 rounded-xl transition-all duration-300 text-center cursor-pointer ${
              isDragging
                ? "scale-105 border-purple-500 bg-gray-800/10"
                : "hover:bg-gray-800/10"
            }`}
          >
            <Upload className="w-12 h-12 mb-4 text-purple-500 mx-auto" />
            <p className="text-lg mb-2">Drag and drop your pool images here</p>
            <p className="text-sm text-gray-400">or</p>
            <button
              onClick={() => document.getElementById("poolUpload").click()}
              className="mt-4 px-6 py-2 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] text-white rounded-md shadow-md text-sm"
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
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
            {poolImages.map((img) => (
              <div
                key={img._id}
                className="relative bg-gray-800 rounded-lg overflow-hidden h-32"
              >
                <img
                  src={img.imageUrl}
                  alt="pool"
                  className="w-full h-full object-cover hover:scale-105 transition"
                />
                <button
                  onClick={() => handleDelete("pool", img._id)}
                  className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 hover:bg-red-700"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Results Section */}
        <div className="w-full text-center">
          <h1 className="text-3xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-pink-500">
            Results
          </h1>
          <div className="bg-gray-800 p-6 rounded-lg text-center">
            <p className="text-lg">Results will be displayed here.</p>
          </div>
        </div>

        {/* Progress Bar */}
        {loading && (
          <div className="w-full max-w-[1048px] mt-8">
            <div className="w-full h-2 bg-gray-800 rounded-full">
              <div
                className="h-full bg-gradient-to-r from-pink-500 to-purple-500 rounded-full animate-pulse"
                style={{ width: "100%" }}
              />
            </div>
          </div>
        )}

        {/* Download Section */}
        <div className="w-full mt-8 text-center">
          <div
            onClick={handleDownload}
            className="flex items-center justify-center px-4 py-2 rounded bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-sm cursor-pointer transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)]"
          >
            <span className="text-gray-200 mr-2">Download</span>
            <img
              src={downloadIcon}
              className="w-[9.2px] h-[5.7px] rotate-[-90deg]"
              alt="download"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
