// src/components/Login.js

import React, { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import { signInStart, signInSuccess, signInFailure } from "../redux/userSlice";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "react-toastify";
import { FaEye, FaEyeSlash } from "react-icons/fa";

const Login = () => {
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false); // New loading state
    const dispatch = useDispatch();
    const navigate = useNavigate();

    // Redirect if already logged in
    useEffect(() => {
        const token = localStorage.getItem("token");
        if (token) navigate("/dashboard");
    }, [navigate]);

    // Clear token on page unload
    useEffect(() => {
        const handleBeforeUnload = () => {
            localStorage.removeItem("token");
        };
        window.addEventListener("beforeunload", handleBeforeUnload);
        return () => window.removeEventListener("beforeunload", handleBeforeUnload);
    }, []);

    // Handle input changes
    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setFormData({ ...formData, [name]: value });
    };

    // Toggle password visibility
    const togglePasswordVisibility = () => {
        setShowPassword((prev) => !prev);
    };

    // Submit login form
const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.email || !formData.password) {
        dispatch(signInFailure("Please fill all fields"));
        toast.error("Please fill all fields");
        return;
    }

    setLoading(true);

    try {
        dispatch(signInStart());
        const response = await fetch(`http://localhost:5000/api/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData),
        });
        const data = await response.json();

        if (response.ok) {
            const { token, user } = data;
            if (token) {
                localStorage.setItem("token", token);
            }
            dispatch(signInSuccess({ user, token }));
            toast.success("Login successful!");
            const dashboardRoute = user.role === 'none' ? `/dashboard/`:`/dashboard/${user.role}`;
            navigate(dashboardRoute);
        } else if (response.status === 401) {
            // Handle invalid credentials specifically
            dispatch(signInFailure(data.message));
            toast.error("Invalid email or password. Please try again.");
        } else {
            // Handle other errors
            dispatch(signInFailure(data.message || "Login failed. Please try again."));
            toast.error(data.message || "Login failed. Please try again.");
        }
    } catch (error) {
        dispatch(signInFailure("An error occurred during sign-in"));
        toast.error("An error occurred during sign-in");
    } finally {
        setLoading(false);
    }
};

    return (
        <div className="flex flex-col lg:flex-row items-stretch lg:h-screen p-3 my-2">
            {/* Left section with image */}
            <div
                className="w-full lg:w-1/2 bg-cover bg-center hidden lg:block"
                style={{ backgroundImage: "url(/images/1.png)", minHeight: "100vh" }}
            ></div>
            {/* Right section with form */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-6 lg:p-0">
                <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-8">
                    <h2 className="text-4xl font-semibold text-navy text-center mb-6">
                        Login
                    </h2>
                    <form onSubmit={handleSubmit}>
                        <div className="space-y-6">
                            <div className="flex flex-col">
                                <label htmlFor="email" className="text-navy font-medium mb-2">Email</label>
                                <input
                                    id="email"
                                    name="email"
                                    type="email"
                                    required
                                    placeholder="Email"
                                    value={formData.email}
                                    onChange={handleInputChange}
                                    className="w-full border-2 border-gray-300 rounded-lg focus:ring-teal-500 focus:border-teal-500 py-2 px-3"
                                />
                            </div>
                            <div className="flex flex-col">
                                <label htmlFor="password" className="text-navy font-medium mb-2">Password</label>
                                <div className="relative">
                                    <input
                                        id="password"
                                        name="password"
                                        type={showPassword ? "text" : "password"}
                                        autoComplete="current-password"
                                        placeholder="Password"
                                        value={formData.password}
                                        onChange={handleInputChange}
                                        className="w-full border-2 border-gray-300 rounded-lg focus:ring-teal-500 focus:border-teal-500 py-2 px-3"
                                    />
                                    <button
                                        type="button"
                                        onClick={togglePasswordVisibility}
                                        aria-label={showPassword ? "Hide password" : "Show password"}
                                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 focus:outline-none"
                                    >
                                        {showPassword ? <FaEyeSlash /> : <FaEye />}
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="mt-8">
                            <button
                                type="submit"
                                className="w-full py-2 bg-teal text-white font-semibold rounded-lg hover:bg-teal-600 focus:ring-teal"
                                disabled={loading} // Disable button while loading
                            >
                                {loading ? "Logging in..." : "Login"}
                            </button>
                        </div>
                    </form>
                    <div className="text-center mt-4">
                        <Link to="/forgotpassword" className="text-teal hover:underline">Forgot Password?</Link>
                    </div>
                    <div className="text-center mt-4">
                        <span className="text-gray">Don't have an account? </span>
                        <Link to="/signup" className="text-teal hover:underline">Sign Up</Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
