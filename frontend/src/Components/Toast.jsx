import React, { useEffect } from 'react';
import { X, AlertCircle, CheckCircle, Info } from 'react-feather';

const Toast = ({ message, type = 'success', onClose, duration = 3000 }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const getToastStyles = () => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-green-500/10',
          border: 'border-green-500/30',
          icon: <CheckCircle size={20} className="text-green-500" />,
          text: 'text-green-400'
        };
      case 'error':
        return {
          bg: 'bg-red-500/10',
          border: 'border-red-500/30',
          icon: <AlertCircle size={20} className="text-red-500" />,
          text: 'text-red-400'
        };
      case 'info':
        return {
          bg: 'bg-blue-500/10',
          border: 'border-blue-500/30',
          icon: <Info size={20} className="text-blue-500" />,
          text: 'text-blue-400'
        };
      default:
        return {
          bg: 'bg-green-500/10',
          border: 'border-green-500/30',
          icon: <CheckCircle size={20} className="text-green-500" />,
          text: 'text-green-400'
        };
    }
  };

  const styles = getToastStyles();

  return (
    <div className="fixed top-5 right-5 z-50 animate-fadeIn">
      <div className={`${styles.bg} ${styles.border} border rounded-lg shadow-lg p-4 max-w-md flex items-start`}>
        <div className="flex-shrink-0 mr-3">
          {styles.icon}
        </div>
        <div className={`flex-1 ${styles.text}`}>
          <p className="text-sm font-medium">{message}</p>
        </div>
        <button 
          onClick={onClose}
          className="ml-4 text-gray-400 hover:text-gray-200 transition-colors"
        >
          <X size={18} />
        </button>
      </div>
    </div>
  );
};

export default Toast;