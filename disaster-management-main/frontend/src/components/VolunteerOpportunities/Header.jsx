import React from "react";

const Header = () => (
  <div className="relative w-full my-5 rounded-lg shadow-lg overflow-hidden">
    {/* Image */}
    <img
      src="/images/volunteer.jpg"
      alt="Volunteer Header"
      className="w-full h-[65vh] object-cover object-top" // Use 75% of the viewport height
    />
    {/* Text below image */}
    <div className="w-full bg-navy-dark text-white py-4 text-center mt-2">
      <h1 className="text-3xl font-bold">Help the people in need ! Volunteer today </h1>
      <p className="text-lg">Join us in making a difference today!</p>
    </div>
  </div>
);

export default Header;
