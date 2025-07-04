import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const [data, setData] = useState(null);
  const [accordionOpen, setAccordionOpen] = useState(0);
  const [showSubscribeModal, setShowSubscribeModal] = useState(false); // state to manage modal visibility
  const [mobileNumber, setMobileNumber] = useState(""); // state for mobile number
  const [otp, setOtp] = useState(""); // state for OTP
  const [otpSent, setOtpSent] = useState(false); // state to track if OTP was sent
  const [errorMessage, setErrorMessage] = useState(""); // state for error messages
  const aboutUsRef = useRef(null);
  const navigate = useNavigate();

  // Fetching homepage data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get("/api/homepage");
        setData(response.data);
      } catch (error) {
        console.error("Error fetching HomePage data:", error);
        // Instead of setting data to null, set to an empty object to allow fallback UI
        setData({});
      }
    };
    fetchData();
  }, []);

  // Adjusting height based on content
  useEffect(() => {
    const adjustHeight = () => {
      if (aboutUsRef.current) {
        const accordionHeight = document.querySelector(".accordion-container")?.scrollHeight || 0;
        const activeDisastersHeight = document.querySelector(".active-disasters-container")?.scrollHeight || 0;
        aboutUsRef.current.style.height = `${Math.max(accordionHeight, activeDisastersHeight, 300)}px`;
      }
    };
    adjustHeight();
  }, [accordionOpen]);

  const toggleAccordion = (index) => {
    setAccordionOpen(index);
  };

  const sendOtpToMobile = async (mobileNumber) => {
    try {
      const response = await axios.post("http://localhost:5000/api/subscribe/send-otp", { mobileNumber });
      return response.data; // { success: true/false, error?: string }
    } catch (error) {
      console.error("Error sending OTP:", error);
      return { success: false, error: "Failed to send OTP. Please try again." };
    }
  };

  const validateOtp = async (mobileNumber, otp) => {
    try {
      const response = await axios.post("http://localhost:5000/api/subscribe/verify-otp", { mobileNumber, otp });
      return response.data; // { success: true/false, error?: string }
    } catch (error) {
      console.error("Error validating OTP:", error);
      return { success: false, error: "Invalid OTP. Please try again." };
    }
  };

  const handleSubscribe = async () => {
    if (otpSent) {
      // Validate OTP
      const result = await validateOtp(mobileNumber, otp);
      if (result.success) {
        alert("Subscription successful!");
        handleModalClose(); // Close the modal on success
      } else {
        setErrorMessage(result.error); // Show error message if OTP validation fails
      }
    } else {
      // Send OTP
      const result = await sendOtpToMobile(mobileNumber);
      if (result.success) {
        setOtpSent(true); // OTP sent, show OTP field
        setErrorMessage(""); // Clear any previous error
      } else {
        setErrorMessage(result.error); // Show error message if OTP sending fails
      }
    }
  };

  // Close modal function
  const handleModalClose = () => {
    setShowSubscribeModal(false); // Close the modal
    setMobileNumber(""); // Reset mobile number
    setOtp(""); // Reset OTP
    setOtpSent(false); // Reset OTP sent flag
    setErrorMessage(""); // Clear error message
  };

  // Provide fallback data for UI sections if data is null or incomplete
  const safeData = {
    hero: data?.hero || { imageUrl: "https://via.placeholder.com/600x300?text=Disaster+Awareness" },
    accordion: Array.isArray(data?.accordion) && data.accordion.length > 0 ? data.accordion : [
      { title: "No Information", links: [{ text: "No links available", url: "#" }] }
    ],
    activeDisasters: Array.isArray(data?.activeDisasters) && data.activeDisasters.length > 0 ? data.activeDisasters : [
      { emoji: "❓", name: "No active disasters" }
    ],
    heroesOfTheDay: Array.isArray(data?.heroesOfTheDay) && data.heroesOfTheDay.length > 0 ? data.heroesOfTheDay : [
      { name: "No Hero", achievement: "No achievements today.", backgroundColor: "#64748b" }
    ],
    mediaGallery: Array.isArray(data?.mediaGallery) && data.mediaGallery.length > 0 ? data.mediaGallery : [],
  };

  return (
    <div className="container mx-auto px-4 py-6 bg-gray-light">
      {/* HERO SECTION */}
      <section className="flex justify-center items-center w-full bg-gradient-to-r from-teal to-navy-light py-12 rounded-lg shadow-md">
        <img
          src={safeData.hero.imageUrl}
          alt="Disaster Awareness"
          className="object-contain max-w-full max-h-full rounded-lg shadow-lg"
        />
      </section>

      {/* ACCORDION, ABOUT US, ACTIVE DISASTERS */}
      <section className="flex flex-wrap lg:flex-nowrap gap-6 px-6 py-16 bg-gray-light">
        {/* Accordion Section */}
        <div className="w-full lg:w-1/4 bg-white p-6 rounded-lg shadow-md transition duration-300 hover:shadow-lg h-[400px] flex flex-col">
          <h2 className="text-2xl font-bold mb-4 text-navy-dark">Information</h2>
          <div className="space-y-4 h-full overflow-y-auto accordion-container">
            {safeData.accordion.map((item, index) => (
              <div
                key={index}
                className={`border rounded transition ${accordionOpen === index ? "border-teal" : "border-gray"}`}
              >
                <button
                  className="w-full text-left px-4 py-2 text-gray font-semibold hover:bg-teal-light transition duration-200"
                  onClick={() => toggleAccordion(index)}
                >
                  {item.title}
                </button>
                {accordionOpen === index && (
                  <div className="px-4 py-2 bg-gray-light text-gray overflow-y-auto max-h-40">
                    <ul className="space-y-2">
                      {item.links.map((link, linkIndex) => (
                        <li key={linkIndex}>
                          <a
                            href={link.url}
                            className="text-teal underline hover:text-teal-dark transition duration-200"
                          >
                            {link.text}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* About Us Section */}
        <div
          ref={aboutUsRef}
          className="w-full lg:w-1/2 bg-white p-6 rounded-lg shadow-md flex flex-col justify-center transition duration-300 hover:shadow-lg"
        >
          <h2 className="text-2xl font-bold mb-4 text-navy-dark">About Us</h2>
          <p className="text-gray leading-relaxed mb-4">
            Our mission is to provide timely and reliable information for disaster management, connect people with
            emergency resources, and help individuals get involved in disaster relief efforts.
          </p>
          <p className="text-gray leading-relaxed">
            Whether you're looking to learn about disasters, find emergency shelters, or contribute to disaster relief
            efforts, our platform provides all the tools you need in one place.
          </p>
        </div>

        {/* Active Disasters Section */}
        <div className="w-full lg:w-1/4 bg-white p-6 rounded-lg shadow-md h-[400px] flex flex-col active-disasters-container transition duration-300 hover:shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-navy-dark">Active Disasters</h2>
          <div className="overflow-hidden h-full border-2 border-teal rounded-lg">
            <motion.div
              animate={{ y: ["0%", "-100%"] }}
              transition={{
                duration: 15,
                ease: "linear",
                repeat: Infinity,
              }}
              className="space-y-4"
            >
              {safeData.activeDisasters.map((disaster, index) => (
                <p key={index} className="text-gray">
                  {disaster.emoji} {disaster.name}
                </p>
              ))}
            </motion.div>
          </div>
          <div>
            <p className="text-gray mt-4">
              Sign up for SMS alerts to receive critical updates during active disasters or emergency situations.
            </p>
            <button
              className="mt-4 bg-teal text-white px-6 py-2 rounded shadow hover:bg-teal-dark transition duration-200"
              onClick={() => setShowSubscribeModal(true)}
            >
              Subscribe for Updates
            </button>
          </div>
        </div>

        {/* Modal for subscribing */}
        {showSubscribeModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-lg max-w-sm w-full">
              <h3 className="text-xl font-bold text-navy-dark mb-4">Subscribe for SMS Updates</h3>
              <input
                type="text"
                value={mobileNumber}
                onChange={(e) => setMobileNumber(e.target.value)}
                placeholder="Enter your mobile number"
                className="w-full px-4 py-2 border rounded-md mb-4"
              />
              {otpSent && (
                <input
                  type="text"
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  placeholder="Enter OTP"
                  className="w-full px-4 py-2 border rounded-md mb-4"
                />
              )}
              {errorMessage && <p className="text-red-500 text-sm">{errorMessage}</p>}
              <button
                onClick={handleSubscribe}
                className="w-full bg-teal text-white py-2 rounded-md mt-4 hover:bg-teal-dark transition duration-200"
              >
                {otpSent ? "Verify OTP" : "Send OTP"}
              </button>
              <button
                onClick={handleModalClose}
                className="w-full mt-4 bg-gray-300 text-navy-dark py-2 rounded-md hover:bg-gray-400 transition duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="px-6 py-16 bg-gray-light shadow-md transition duration-300 hover:shadow-lg">
        <h2 className="text-4xl font-bold mb-8 text-center text-navy-dark">
          Our Features
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            {
              title: "Disaster Feed Page",
              description:
                "Stay up to date with real-time information about active disasters. Our interactive map allows you to view disaster locations, check if you are in a red zone, and explore data on past and present disasters.",
              path: "/disaster-feed",  // Path for the first card
            },
            {
              title: "Emergency Center Locator",
              description:
                "Quickly find emergency centers near you, including hospitals, shelters, and other critical resources. Access contact information for local coordinators and shelter availability.",
              path: "/shelter",  // Path for the second card
            },
            {
              title: "Disaster Awareness",
              description:
                "Learn about different types of disasters, their causes, effects, and impacts on communities.",
              path: "/awareness/Earthquake",  // Path for the third card
            },
            {
              title: "Disaster Preparedness",
              description:
                "Equip yourself with essential knowledge and skills to prepare for any emergency. We provide resources to help individuals and communities be ready before disaster strikes.",
              path: "/preparedness/Earthquake",  // Path for the fourth card
            },
          ].map((feature, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-lg shadow-md transition-all duration-200 hover:shadow-lg hover:scale-102 flex flex-col justify-between"
            >
              <div>
                <h3 className="text-xl font-bold mb-4 text-navy-dark">
                  {feature.title}
                </h3>
                <p className="text-gray mb-4">{feature.description}</p>
              </div>
              <div className="mt-auto flex justify-center">
                <button
                  className="bg-teal text-white px-4 py-2 rounded transition-all duration-300 hover:bg-teal-dark hover:scale-105"
                  onClick={() => navigate(feature.path)}  // Use dynamic path for navigation
                >
                  Learn More
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>


      <section className="px-6 py-16 bg-gradient-to-r from-navy-light to-teal-light text-center shadow-md transition duration-300 hover:shadow-lg relative z-10">
        <h2 className="text-4xl font-bold mb-8" style={{ color: "#1E293B" }}>
          Get Involved
        </h2>
        <p className="leading-relaxed mb-4" style={{ color: "#333333" }}>
          Contribute to disaster relief efforts by volunteering in various capacities, such as rescue operations or public
          awareness campaigns. You can also make a financial donation or donate rations to help those affected by disasters.
          Every contribution counts!
        </p>
        <div className="mt-6 flex justify-center gap-4">
          <button
            className="bg-green text-white px-6 py-3 rounded shadow hover:bg-green-dark transition duration-300 transform hover:scale-105"
            onClick={() => navigate('/volunteer')} // Navigate to the Get Involved page
          >
            Volunteer
          </button>
          <button
            className="bg-yellow text-white px-6 py-3 rounded shadow hover:bg-red transition duration-300 transform hover:scale-105"
            onClick={() => navigate('/donate')} // Navigate to the Get Involved page
          >
            Donate
          </button>
        </div>
      </section>



      {/* HEROES OF THE DAY */}
      <section className="my-16 text-center">
        <h2 className="text-4xl font-bold mb-8 text-navy-dark">Heroes of the Day</h2>
        <div className="flex justify-center gap-6">
          {safeData.heroesOfTheDay.map((hero, index) => (
            <div
              key={index}
              className="text-white p-6 rounded-lg shadow-md w-72 transition duration-300 hover:shadow-lg hover:scale-105"
              style={{ backgroundColor: hero.backgroundColor }}
            >
              <h3 className="text-xl font-bold mb-2">🏅 {hero.name}</h3>
              <p>{hero.achievement}</p>
            </div>
          ))}
        </div>
      </section>

      {/* MEDIA GALLERY */}
      <section className="px-6 py-16 bg-gray-light">
        <h2 className="text-4xl font-bold mb-8 text-center text-navy-dark">Media Gallery</h2>
        <div className="flex justify-center gap-6">
          {/* Image Gallery */}
          <div className="bg-white p-6 rounded-lg shadow-md overflow-y-scroll h-96 w-[40%] transition duration-300 hover:shadow-lg">
            <h3 className="text-2xl font-bold mb-4 text-navy-dark">Image Gallery</h3>
            {safeData.mediaGallery.filter((media) => media.type === "image").length > 0 ? (
              safeData.mediaGallery
                .filter((media) => media.type === "image")
                .map((media, index) => (
                  <div key={index} className="mb-4">
                    <img
                      src={media.content}
                      alt={`Image ${index + 1}`}
                      className="w-full rounded shadow-md transition duration-200 hover:shadow-lg"
                    />
                    {media.description && <p className="mt-2 text-gray">{media.description}</p>}
                  </div>
                ))
            ) : (
              <p className="text-gray">No images available.</p>
            )}
          </div>

          {/* Video Gallery */}
          <div className="bg-white p-6 rounded-lg shadow-md overflow-y-scroll h-96 w-[40%] transition duration-300 hover:shadow-lg">
            <h3 className="text-2xl font-bold mb-4 text-navy-dark">Video Gallery</h3>
            {safeData.mediaGallery.filter((media) => media.type === "video").length > 0 ? (
              safeData.mediaGallery
                .filter((media) => media.type === "video")
                .map((media, index) => (
                  <div key={index} className="mb-4">
                    <video controls className="w-full rounded shadow-md transition duration-200 hover:shadow-lg">
                      <source src={media.content} type="video/mp4" />
                    </video>
                    {media.description && <p className="mt-2 text-gray">{media.description}</p>}
                  </div>
                ))
            ) : (
              <p className="text-gray">No videos available.</p>
            )}
          </div>
        </div>
      </section>

    </div>
  );
};

export default HomePage;
