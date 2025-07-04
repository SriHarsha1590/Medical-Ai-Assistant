import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-navy text-white py-4 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row justify-between items-center">
          <p className="text-center sm:text-left">
            &copy; {new Date().getFullYear()} Disaster Management Website. All rights reserved.
          </p>
          <div className="mt-2 sm:mt-0 flex space-x-4">
            <a href="/privacy" className="text-white hover:text-teal">Privacy Policy</a>
            <a href="/terms" className="text-white hover:text-teal">Terms of Service</a>
            <a href="/contact" className="text-white hover:text-teal">Contact Us</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
