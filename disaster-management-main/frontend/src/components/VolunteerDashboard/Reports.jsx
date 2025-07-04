// src/components/Reports.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAppContext } from "../context/AppContext";

const Reports = () => {
  const { user } = useAppContext();
  const [reports, setReports] = useState([]);
  const [newReport, setNewReport] = useState({
    title: "",
    description: "",
  });

  useEffect(() => {
    if (user) {
      axios
        .get(`/api/reports/${user.email}`)
        .then((response) => setReports(response.data))
        .catch((error) => console.error("Error fetching reports:", error));
    }
  }, [user]);

  const handleInputChange = (e) => {
    setNewReport({ ...newReport, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    axios
      .post(`/api/reports/${user.email}`, newReport)
      .then((response) => {
        setReports([...reports, response.data]);
        setNewReport({ title: "", description: "" });
      })
      .catch((error) => console.error("Error submitting report:", error));
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-navy mb-4">Reports</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          name="title"
          value={newReport.title}
          onChange={handleInputChange}
          placeholder="Report Title"
          className="p-2 w-full border rounded"
        />
        <textarea
          name="description"
          value={newReport.description}
          onChange={handleInputChange}
          placeholder="Report Description"
          className="p-2 w-full border rounded"
        />
        <button type="submit" className="bg-teal text-white p-2 rounded">Submit Report</button>
      </form>

      <h3 className="text-xl font-bold text-navy mt-6">Submitted Reports</h3>
      {reports.length > 0 ? (
        <ul>
          {reports.map((report) => (
            <li key={report._id} className="bg-gray-light p-4 my-2 rounded">
              <h4 className="font-bold">{report.title}</h4>
              <p>{report.description}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p>No reports submitted yet.</p>
      )}
    </div>
  );
};

export default Reports;
