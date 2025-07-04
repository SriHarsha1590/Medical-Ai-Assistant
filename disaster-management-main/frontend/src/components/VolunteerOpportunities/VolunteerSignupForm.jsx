import React, { useState } from "react";
import { apiClient } from "../../api/apiservice";

const VolunteerSignupForm = ({ opportunityId }) => {
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const handleSignup = async () => {
    setLoading(true);
    try {
        console.log('signing up');
      await apiClient.post(`/volunteer-opportunities/${opportunityId}/signup`, {
        userEmail :state.user.currentUser.email,
      });
      setSuccessMessage("You have successfully signed up!");
    } catch (error) {
      alert("Error signing up. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="signup-form mt-4">
      {successMessage ? (
        <p className="text-green-500">{successMessage}</p>
      ) : (
        <button
          onClick={handleSignup}
          disabled={loading}
          className="bg-green-500 text-white px-4 py-2 rounded"
        >
          {loading ? "Signing Up..." : "Confirm Signup"}
        </button>
      )}
    </div>
  );
};

export default VolunteerSignupForm;
