import React, { useState } from "react";
import axios from "axios";

const BASE_URL = "http://localhost:5000/api"; // Replace with your backend base URL

const Donation = ({ onDonationUpdate }) => {
  // State for form fields
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    amount: 500, // Default amount set to ₹500
    isRecurring: false,
    message: "",
  });

  // State for loading and errors
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  // Handle form submission
  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      // Step 1: Create an order
      const orderResponse = await axios.post(`${BASE_URL}/donations/create-order`, {
        amount: formData.amount,
        isRecurring: formData.isRecurring,
        name: formData.name,
        email: formData.email,
        message: formData.message,
      });
      const { id } = orderResponse.data;

      // Step 2: Redirect to PayPal for payment approval (assuming PayPal integration supports INR)
      window.location.href = `https://www.sandbox.paypal.com/checkoutnow?token=${id}`;
    } catch (err) {
      setError("There was an error processing your donation. Please try again.");
      console.error("Error creating order:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Step 3: Capture payment after PayPal redirect
  const handleCapturePayment = async (orderId) => {
    try {
      const captureResponse = await axios.post(`${BASE_URL}/donations/capture-order/${orderId}`);
      console.log("Payment captured successfully:", captureResponse.data);

      // Show success message or perform additional actions
      alert("Thank you for your donation!");
      onDonationUpdate(captureResponse.data);
    } catch (err) {
      console.error("Error capturing payment:", err);
      alert("Failed to capture payment. Please contact support.");
    }
  };

  // Check URL for PayPal redirect and capture payment
  React.useEffect(() => {
    const queryParams = new URLSearchParams(window.location.search);
    const orderId = queryParams.get("token");

    if (orderId) {
      handleCapturePayment(orderId);
    }
  }, []);

  return (
    <div className="donation-section flex flex-col md:flex-row items-center justify-between bg-teal-light p-6 rounded-lg shadow-lg">
      {/* Left: Image */}
      <div className="donation-image md:w-1/2 mb-4 md:mb-0 md:mr-6">
        <img
          src="/images/donation.avif"
          alt="Donate Money Impact"
          className="rounded-lg shadow-md w-full h-auto object-cover"  // Ensures image covers the space without distortion
        />
      </div>

      {/* Right: Content */}
      <div className="donation-content md:w-1/2 flex flex-col">
        <h2 className="text-2xl font-bold text-navy mb-4">
          Make a Difference with Your Donation
        </h2>
        <p className="text-gray mb-6">
          Your contribution helps provide essential resources to those in need.
          Every Rupee counts!
        </p>

        {/* Suggested Donation Amounts */}
        <div className="suggested-amounts flex gap-4 mb-6">
          {[500, 2500, 5000].map((amount) => (
            <button
              key={amount}
              className={`px-4 py-2 rounded-md bg-teal text-white font-medium hover:bg-teal-700 ${formData.amount === amount ? "ring-2 ring-yellow" : ""
                }`}
              onClick={() => setFormData({ ...formData, amount })}
            >
              ₹{amount}
            </button>
          ))}
        </div>


        {/* Recurring Donation Toggle */}
        <label className="flex items-center gap-2 ">
          <input
            type="checkbox"
            name="isRecurring"
            checked={formData.isRecurring}
            onChange={handleInputChange}
            className="form-checkbox text-teal rounded-md"
          />
          <span className="text-gray">
            Make this a <strong>monthly recurring</strong> donation.
          </span>
        </label>
        {/* Added Info */}
        <p className="text-sm text-gray ml-6 mb-6">
          <strong>You can cancel it anytime.</strong>
        </p>


        {/* Donation Form */}
        <form onSubmit={handleFormSubmit} className="flex flex-col gap-4">
          <input
            type="text"
            name="name"
            placeholder="Your Name"
            value={formData.name}
            onChange={handleInputChange}
            className="border-gray-light border p-2 rounded-md"
            required
          />
          <input
            type="email"
            name="email"
            placeholder="Your Email"
            value={formData.email}
            onChange={handleInputChange}
            className="border-gray-light border p-2 rounded-md"
            required
          />
          <textarea
            name="message"
            placeholder="Leave a message (optional)"
            value={formData.message}
            onChange={handleInputChange}
            className="border-gray-light border p-2 rounded-md"
          />
          <button
            type="submit"
            className={`bg-green text-white px-6 py-2 rounded-md ${isLoading ? "opacity-50 cursor-not-allowed" : "hover:bg-teal-dark"
              }`}
            disabled={isLoading}
          >
            {isLoading ? "Processing..." : "Donate Now"}
          </button>
        </form>

        {/* Error Message */}
        {error && <p className="text-red mt-4">{error}</p>}
      </div>
    </div>
  );
};

export default Donation;
