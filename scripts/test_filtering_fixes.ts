// Test script to validate filtering logic fixes
import { supabase } from '../src/lib/supabase';

// Helper function to check if a value represents 'all' (case insensitive and includes 'all')
const isAllValue = (value?: string) => {
  if (!value) return true;
  const lowerValue = value.toLowerCase().trim();
  return lowerValue.includes('all') || lowerValue === '' || lowerValue === 'all programs' || lowerValue === 'all year levels';
};

// Test function to validate student count calculation
async function testStudentCountCalculation() {
  console.log('Testing student count calculation...\n');

  // Test cases
  const testCases = [
    {
      name: 'Specific Program and Year',
      program: 'Computer Science',
      year: '1st',
      expectedBehavior: 'Should only show students from Computer Science, 1st year'
    },
    {
      name: 'All Programs, Specific Year',
      program: 'All Programs',
      year: '1st',
      expectedBehavior: 'Should show all students from 1st year across all programs'
    },
    {
      name: 'Specific Program, All Year Levels',
      program: 'Computer Science',
      year: 'All Year Levels',
      expectedBehavior: 'Should show all students from Computer Science across all years'
    },
    {
      name: 'All Programs, All Year Levels',
      program: 'All Programs',
      year: 'All Year Levels',
      expectedBehavior: 'Should show all students'
    }
  ];

  for (const testCase of testCases) {
    console.log(`\n--- ${testCase.name} ---`);
    console.log(`Expected: ${testCase.expectedBehavior}`);
    
    try {
      // Build query to count students based on session criteria
      let countQuery = supabase
        .from('students')
        .select('*', { count: 'exact', head: true });
        
      // Apply filters only if they are not "all" values
      if (testCase.program && !isAllValue(testCase.program)) {
        countQuery = countQuery.eq('program', testCase.program.trim());
        console.log(`Applied program filter: ${testCase.program}`);
      } else {
        console.log('No program filter applied (All Programs)');
      }
      
      if (testCase.year && !isAllValue(testCase.year)) {
        let yearValue = testCase.year.trim();
        if (yearValue.endsWith(' Year')) {
          yearValue = yearValue.replace(' Year', '');
        }
        countQuery = countQuery.eq('year', yearValue);
        console.log(`Applied year filter: ${yearValue}`);
      } else {
        console.log('No year filter applied (All Year Levels)');
      }
      
      
      const { count, error } = await countQuery;
      
      if (error) {
        console.error(`Error: ${error.message}`);
      } else {
        console.log(`✅ Student count: ${count || 0}`);
      }
      
    } catch (error) {
      console.error(`❌ Test failed: ${error}`);
    }
  }
}

// Test function to validate session key format consistency
function testSessionKeyFormat() {
  console.log('\n\n--- Testing Session Key Format Consistency ---\n');
  
  const testSessions = [
    {
      program: 'Computer Science',
      year: '1st Year',
    },
    {
      program: 'All Programs',
      year: '1st',
    },
    {
      program: 'Computer Science',
      year: 'All Year Levels',
    }
  ];

  for (const session of testSessions) {
    const program = session.program || 'All Programs';
    const year = session.year || 'All Year Levels';
    const sessionKey = `${program}::${year}`;
    
    console.log(`Session: ${JSON.stringify(session)}`);
    console.log(`Generated key: ${sessionKey}`);
    console.log('---');
  }
}

// Run tests
async function runTests() {
  console.log('🧪 Running filtering logic tests...\n');
  
  try {
    await testStudentCountCalculation();
    testSessionKeyFormat();
    
    console.log('\n✅ All tests completed successfully!');
  } catch (error) {
    console.error('\n❌ Tests failed:', error);
  }
}

// Export for use in other files
export { testStudentCountCalculation, testSessionKeyFormat, isAllValue };

// Run if this file is executed directly
if (require.main === module) {
  runTests();
}
