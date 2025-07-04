// src/components/VolunteerProfile.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import { useAppContext } from '.../context/AppContext'

const VolunteerProfile = () => {
  const { user } = useAppContext();
  const [profile, setProfile] = useState(null);
  const [editing, setEditing] = useState(false);
  const [formData, setFormData] = useState({
    fullName: "",
    city: "",
    area: "",
    profession: "",
    mobileNumber: "",
    availabilityDropdown: "",
  });

  useEffect(() => {
    if (user) {
      axios
        .get(`/api/volunteer-profile/${user.email}`)
        .then((response) => setProfile(response.data))
        .catch((error) => console.error("Error fetching profile:", error));
    }
  }, [user]);

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    axios
      .put(`/api/volunteer-profile/${user.email}`, formData)
      .then((response) => {
        setProfile(response.data);
        setEditing(false);
      })
      .catch((error) => console.error("Error updating profile:", error));
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-navy mb-4">Volunteer Profile</h2>
      {profile ? (
        <div>
          {editing ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              <input
                type="text"
                name="fullName"
                value={formData.fullName}
                onChange={handleInputChange}
                placeholder="Full Name"
                className="p-2 w-full border rounded"
              />
              <input
                type="text"
                name="city"
                value={formData.city}
                onChange={handleInputChange}
                placeholder="City"
                className="p-2 w-full border rounded"
              />
              <input
                type="text"
                name="area"
                value={formData.area}
                onChange={handleInputChange}
                placeholder="Area"
                className="p-2 w-full border rounded"
              />
              <input
                type="text"
                name="profession"
                value={formData.profession}
                onChange={handleInputChange}
                placeholder="Profession"
                className="p-2 w-full border rounded"
              />
              <input
                type="text"
                name="mobileNumber"
                value={formData.mobileNumber}
                onChange={handleInputChange}
                placeholder="Mobile Number"
                className="p-2 w-full border rounded"
              />
              <input
                type="text"
                name="availabilityDropdown"
                value={formData.availabilityDropdown}
                onChange={handleInputChange}
                placeholder="Availability"
                className="p-2 w-full border rounded"
              />
              <button type="submit" className="bg-teal text-white p-2 rounded">Save</button>
            </form>
          ) : (
            <div className="space-y-4">
              <p><strong>Full Name:</strong> {profile.fullName}</p>
              <p><strong>City:</strong> {profile.city}</p>
              <p><strong>Area:</strong> {profile.area}</p>
              <p><strong>Profession:</strong> {profile.profession}</p>
              <p><strong>Mobile Number:</strong> {profile.mobileNumber}</p>
              <p><strong>Availability:</strong> {profile.availabilityDropdown}</p>
              <button
                onClick={() => {
                  setEditing(true);
                  setFormData(profile);
                }}
                className="bg-yellow text-black p-2 rounded"
              >
                Edit Profile
              </button>
            </div>
          )}
        </div>
      ) : (
        <p>Loading profile...</p>
      )}
    </div>
  );
};

export default VolunteerProfile;
