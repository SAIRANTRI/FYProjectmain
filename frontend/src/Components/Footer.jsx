import { GitHub, Mail, Heart } from 'react-feather';

const Footer = () => {
  return (
    <div className="border-t border-opacity-10 border-white py-6 px-4 w-full bg-black bg-opacity-30 mt-auto">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="flex flex-col">
            <h3 className="text-white font-medium text-lg mb-3">Product</h3>
            <a href="/" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">Home</a>
            <a href="/upload" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">Upload</a>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">FAQ</a>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm transition-all duration-200 hover:translate-x-1 transform inline-block">Documentation</a>
          </div>
          
          <div className="flex flex-col">
            <h3 className="text-white font-medium text-lg mb-3">Team</h3>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">Khush</a>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">Sairantri</a>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 hover:translate-x-1 transform inline-block">Pratik</a>
            <a href="#" className="text-white opacity-60 hover:opacity-100 text-sm transition-all duration-200 hover:translate-x-1 transform inline-block">Atreyee</a>
          </div>
          
          <div className="flex flex-col">
            <h3 className="text-white font-medium text-lg mb-3">Connect</h3>
            <a href="https://github.com/It-is-KD/FYProject" target="_blank" rel="noopener noreferrer" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 flex items-center group">
              <GitHub size={16} className="mr-2 group-hover:text-purple-400 transition-colors" />
              <span className="group-hover:translate-x-1 transition-transform">GitHub</span>
            </a>
            <a href="mailto:contact@albumify.com" className="text-white opacity-60 hover:opacity-100 text-sm mb-2 transition-all duration-200 flex items-center group">
              <Mail size={16} className="mr-2 group-hover:text-purple-400 transition-colors" />
              <span className="group-hover:translate-x-1 transition-transform">Contact Us</span>
            </a>
          </div>
        </div>
        
        <div className="mt-8 pt-4 border-t border-gray-800 text-center">
          <p className="text-white opacity-60 text-sm flex items-center justify-center">
            Made by Team Albumify
          </p>
          <p className="text-white opacity-40 text-xs mt-1">© {new Date().getFullYear()} Albumify. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
};

export default Footer;