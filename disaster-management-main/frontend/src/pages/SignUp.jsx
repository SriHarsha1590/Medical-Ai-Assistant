import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { BASE_URL } from "../api/apiservice";
import { toast } from "react-toastify";
import { Formik, Form, Field } from "formik";
import * as Yup from "yup";
import AdditionalDetailsForm from "./AdditionalDetailsForm";

const SignUp = () => {
  const navigate = useNavigate();
  const [submitted, setSubmitted] = useState(false);
  const [formValues, setFormValues] = useState({});

  const initialValues = {
    username: "",
    email: "",
    password: "",
    role: "none", // Default role is "None"
  };

  const validationSchema = Yup.object({
    username: Yup.string().required("Username is required"),
    email: Yup.string().email("Invalid email address").required("Email is required"),
    password: Yup.string().required("Password is required"),
    role: Yup.string().oneOf(["none", "volunteer", "coordinator", "admin"]).required("Role is required"),
  });

  const handleSubmit = (values) => {
    setFormValues(values);
    if (values.role === "none") {
      handleSignupSubmit({}); // No additional details required for "none"
    } else {
      setSubmitted(true); // Show additional details form
    }
  };

  const handleBack = () => {
    setSubmitted(false);
  };

  const handleSignupSubmit = async (additionalValues) => {
    const formData = { ...formValues, ...additionalValues };
    try {
      const response = await fetch(`${BASE_URL}/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Registration failed");
      }
      navigate("/login");
      toast.success("Registration successful! Please login.");
    } catch (error) {
      console.error("Registration error:", error);
      toast.error(error.message);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row items-stretch mt-10">
      <div
        className="w-full h-screen lg:w-1/2 bg-center lg:block hidden"
        style={{
          backgroundImage: "url(/images/1.png)",
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          minHeight: "100vh",
          height: "auto",
        }}
      ></div>
      <div className="w-full lg:w-1/2 flex items-center justify-center">
        <div className="max-w-md w-full mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
          <h2 className="text-3xl font-extrabold text-gray-900 text-center py-4">
            Sign Up
          </h2>
          {submitted ? (
            <AdditionalDetailsForm
              onBack={handleBack}
              onSubmit={handleSignupSubmit}
              initialValues={{
                fullName: "",
                city: "",
                area: "",
                profession: "",
                mobileNumber: "",
                availabilityDropdown: "",
              }}
            />
          ) : (
            <Formik
              initialValues={initialValues}
              validationSchema={validationSchema}
              onSubmit={handleSubmit}
              validateOnBlur={false}
            >
              {({ errors, touched }) => (
                <Form className="px-8 py-6">
                  <div className="space-y-4">
                    {/* Username Input */}
                    <div className="mb-4">
                      <label
                        htmlFor="username"
                        className="block text-gray-700 font-medium mb-2"
                      >
                        Username
                      </label>
                      <Field
                        id="username"
                        name="username"
                        type="text"
                        placeholder="Username"
                        className="w-full border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 py-2 px-4"
                      />
                      {errors.username && touched.username && (
                        <div className="text-red-500 text-sm">{errors.username}</div>
                      )}
                    </div>

                    {/* Email Input */}
                    <div className="mb-4">
                      <label
                        htmlFor="email"
                        className="block text-gray-700 font-medium mb-2"
                      >
                        Email address
                      </label>
                      <Field
                        id="email"
                        name="email"
                        type="email"
                        placeholder="Email address"
                        className="w-full border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 py-2 px-4"
                      />
                      {errors.email && touched.email && (
                        <div className="text-red-500 text-sm">{errors.email}</div>
                      )}
                    </div>

                    {/* Password Input */}
                    <div className="mb-4">
                      <label
                        htmlFor="password"
                        className="block text-gray-700 font-medium mb-2"
                      >
                        Password
                      </label>
                      <Field
                        id="password"
                        name="password"
                        type="password"
                        placeholder="Password"
                        className="w-full border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 py-2 px-4"
                      />
                      {errors.password && touched.password && (
                        <div className="text-red-500 text-sm">{errors.password}</div>
                      )}
                    </div>

                    {/* Role Select */}
                    <div className="mb-4">
                      <label
                        htmlFor="role"
                        className="block text-gray-700 font-medium mb-2"
                      >
                        Role
                      </label>
                      <Field
                        id="role"
                        name="role"
                        as="select"
                        className="w-full border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 py-2 px-4"
                      >
                        <option value="none">None</option>
                        <option value="volunteer">Volunteer</option>
                        <option value="coordinator">Coordinator</option>
                        <option value="admin">Admin</option>
                      </Field>
                      {errors.role && touched.role && (
                        <div className="text-red-500 text-sm">{errors.role}</div>
                      )}
                    </div>
                  </div>

                  <div className="text-right mt-6">
                    <button
                      type="submit"
                      className="w-full py-2 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      Next
                    </button>
                  </div>
                </Form>
              )}
            </Formik>
          )}

          {/* Sign In Link */}
          <div className={`text-center my-4 ${submitted ? "hidden" : ""}`}>
            <span className="text-gray-700">Have an account? </span>
            <Link to="/login" className="text-blue-500 hover:underline">
              Log in
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
