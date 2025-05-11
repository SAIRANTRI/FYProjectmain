const Footer = () => {
    return (
      <div className="border-t border-opacity-10 border-white flex flex-row justify-around py-4 px-2 w-full bg-black bg-opacity-30 mt-auto">
        <div className="flex flex-col box-border">
          <div className="text-white font-medium text-sm mb-2">Product</div>
          <div className="text-white opacity-60 text-sm mb-1">Home</div>
          <div className="text-white opacity-60 text-sm mb-1">Download</div>
          <span className="text-white opacity-60 text-sm">FAQ</span>
        </div>
        <div className="flex flex-col items-center box-border">
          <div className="text-white font-medium text-sm mb-2">About</div>
          <div className="text-white opacity-60 text-sm mb-1">Khush</div>
          <div className="text-white opacity-60 text-sm mb-1">Sairantri</div>
          <div className="text-white opacity-60 text-sm mb-1">Pratik</div>
          <span className="text-white opacity-60 text-sm">Atreyee</span>
        </div>
      </div>
    );
  };
  
  export default Footer;