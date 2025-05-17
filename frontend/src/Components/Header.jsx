import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuthStore } from "../store/useAuthStore";
import close from '../assets/close.svg';
import logo from '../assets/logo.png';
import menu from '../assets/menu.svg';
import { User, Upload, Home, LogIn, UserPlus } from 'react-feather';

const Header = () => {
  const [toggle, setToggle] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated } = useAuthStore();

  const currentPath = location.pathname;
  const getNavItems = () => {
    const baseItems = [
      { title: "Home", path: "/", icon: <Home size={18} /> },
    ];
    
    if (isAuthenticated) {
      return [
        ...baseItems,
        { title: "Upload", path: "/upload", icon: <Upload size={18} /> },
        { title: "Profile", path: "/profile", icon: <User size={18} /> },
      ];
    } else {
      return [
        ...baseItems,
        { title: "Login", path: "/login", icon: <LogIn size={18} /> },
        { title: "Sign Up", path: "/signup", icon: <UserPlus size={18} /> },
      ];
    }
  };

  const navItems = getNavItems();

  const handleNavigation = (path) => {
    navigate(path);
    setToggle(false);
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (toggle && !e.target.closest('.sidebar') && !e.target.closest('.menu-button')) {
        setToggle(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [toggle]);

  return (
    <nav className="w-full flex py-6 justify-between items-center navbar">
      <div 
        className="flex items-center cursor-pointer" 
        onClick={() => navigate('/')}
      >
        <img
          src={logo}
          alt="Albumify"
          className="h-[25px] transition-transform duration-300 hover:scale-105"
        />
      </div>

      <ul className="list-none sm:flex hidden justify-end items-center flex-1">
        {navItems.map(({ title, path, icon }, index) => (
          <li
            key={title}
            className={`font-poppins font-normal cursor-pointer text-[16px] flex items-center ${
              currentPath === path 
                ? "text-white bg-purple-500/20 rounded-md px-3 py-1.5" 
                : "text-dimWhite hover:text-white hover:bg-purple-500/10 rounded-md px-3 py-1.5"
            } ${index !== navItems.length - 1 ? "mr-2" : ""} transition-all duration-300`}
            onClick={() => handleNavigation(path)}
          >
            <span className="mr-1.5">{icon}</span>
            {title}
          </li>
        ))}
      </ul>

      <div className="sm:hidden flex flex-1 justify-end items-center">
        <img
          src={toggle ? close : menu}
          alt="menu"
          className="w-[28px] h-[28px] object-contain cursor-pointer menu-button"
          onClick={() => setToggle((prev) => !prev)}
        />

        <div
          className={`${
            !toggle ? "hidden" : "flex"
          } p-6 bg-black-gradient absolute top-20 right-0 mx-4 my-2 min-w-[140px] rounded-xl sidebar z-50 bg-gray-900/95 backdrop-blur-lg border border-gray-800 shadow-xl transition-all duration-300 animate-fadeIn`}
        >
          <ul className="list-none flex justify-end items-start flex-1 flex-col">
            {navItems.map(({ title, path, icon }, index) => (
              <li
                key={title}
                className={`font-poppins font-medium cursor-pointer text-[16px] flex items-center w-full ${
                  currentPath === path 
                    ? "text-white bg-purple-500/20 rounded-md px-3 py-2" 
                    : "text-dimWhite hover:text-white hover:bg-purple-500/10 rounded-md px-3 py-2"
                } ${index !== navItems.length - 1 ? "mb-2" : ""} transition-all duration-300`}
                onClick={() => handleNavigation(path)}
              >
                <span className="mr-2">{icon}</span>
                {title}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Header;