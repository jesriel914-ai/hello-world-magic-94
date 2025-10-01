import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import type { Student } from '@/types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Formats student display with student_id and full name
 * Example: "22S-326 - John Doe"
 */
export function formatStudentDisplay(student: Student): string {
  return `${student.student_id} - ${student.firstname} ${student.surname}`;
}

/**
 * Gets student full name (firstname + surname)
 */
export function getStudentFullName(student: Student): string {
  return `${student.firstname} ${student.surname}`;
}

/**
 * Cleans IP address by removing IPv6-mapped IPv4 prefix (::ffff:)
 * Example: "::ffff:192.168.1.1" becomes "192.168.1.1"
 */
export function cleanIpAddress(ipAddress: string): string {
  return ipAddress.replace(/^::ffff:/, '');
}
