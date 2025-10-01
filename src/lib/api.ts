import axios from 'axios';

// Function to get API base URL with fallback for browser environment
const getApiBaseUrl = () => {
  // Check if we're in a browser environment
  if (typeof window !== 'undefined') {
    // In browser, use window.ENV or fallback to localhost
    return window.ENV?.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000/api';
  }
  // In server environment, use process.env
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000/api';
};

const API_BASE_URL = getApiBaseUrl(); // Configurable API base URL

export interface Session {
  id: number;
  title: string;
  type: 'class' | 'event' | 'other';
  time: string;
  location: string;
  instructor: string;
  students: number;
  program: string;
  year: string;
  section: string;
  description: string;
  capacity: string;
  date: string;
}

export const fetchSessions = async (startDate: string, endDate: string): Promise<Session[]> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/sessions`, {
      params: {
        startDate,
        endDate
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching sessions:', error);
    return [];
  }
};

export const createSession = async (sessionData: Omit<Session, 'id'>): Promise<Session> => {
  try {
    const response = await axios.post(`${API_BASE_URL}/sessions`, sessionData);
    return response.data;
  } catch (error) {
    console.error('Error creating session:', error);
    throw error;
  }
};

export const updateSession = async (id: number, sessionData: Partial<Session>): Promise<Session> => {
  try {
    const response = await axios.put(`${API_BASE_URL}/sessions/${id}`, sessionData);
    return response.data;
  } catch (error) {
    console.error('Error updating session:', error);
    throw error;
  }
};

export const deleteSession = async (id: number): Promise<void> => {
  try {
    await axios.delete(`${API_BASE_URL}/sessions/${id}`);
  } catch (error) {
    console.error('Error deleting session:', error);
    throw error;
  }
};
