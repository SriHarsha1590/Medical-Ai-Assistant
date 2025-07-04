import React from 'react';

// Button component with customizable styles and functionality
const Button = ({ type = "button", onClick, children, variant = "primary", size = "medium" }) => {
  const baseStyles = "font-semibold rounded-md focus:outline-none transition duration-300 ease-in-out";

  // Define styles based on the variant and size using your Tailwind config colors
  const variantStyles = {
    primary: "bg-navy text-white hover:bg-teal-light hover:text-navy focus:ring-2 focus:ring-teal-light", // Navy background with teal-light hover and text change
    secondary: "bg-gray-light text-navy hover:bg-gray hover:text-white focus:ring-2 focus:ring-gray", // Light gray with navy text and dark gray hover
    danger: "bg-red text-white hover:bg-red-700 hover:text-white focus:ring-2 focus:ring-red-500", // Red with darker hover and no text change
    success: "bg-teal text-white hover:bg-navy hover:text-white focus:ring-2 focus:ring-teal-light", // Teal with navy hover and white text
  };

  const sizeStyles = {
    small: "px-3 py-1 text-sm",
    medium: "px-5 py-2 text-base",
    large: "px-6 py-3 text-lg",
  };

  return (
    <button
      type={type}
      onClick={onClick}
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]}`}
    >
      {children}
    </button>
  );
};

export default Button;
