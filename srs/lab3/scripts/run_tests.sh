

# Create results directory
mkdir -p results
cd results

echo "Starting performance tests..."

# Test with 2 processes
echo ""
echo ">>> TESTING WITH 2 MPI PROCESSES <<<"
echo "==================================================================="
mpirun -np 2 python3 /Users/richard/parallel-computing-and-multimedia-processing/srs/lab3/mpi_web_scraper.py > test_2_processes.log 2>&1
echo "Test with 2 processes completed. Results saved to test_2_processes.log"

# Test with 3 processes  
echo ""
echo ">>> TESTING WITH 3 MPI PROCESSES <<<"
echo "==================================================================="
mpirun -np 3 python3 /Users/richard/parallel-computing-and-multimedia-processing/srs/lab3/mpi_web_scraper.py > test_3_processes.log 2>&1
echo "Test with 3 processes completed. Results saved to test_3_processes.log"

# Test with 6 processes
echo ""
echo ">>> TESTING WITH 6 MPI PROCESSES <<<"
echo "==================================================================="
mpiexec --oversubscribe -n  6 python3 /Users/richard/parallel-computing-and-multimedia-processing/srs/lab3/mpi_web_scraper.py > test_6_processes.log 2>&1
echo "Test with 6 processes completed. Results saved to test_6_processes.log"

echo ""
echo "==================================================================="
echo "All tests completed! Analyzing results..."
echo "==================================================================="

# Extract timing information
echo ""
echo "PERFORMANCE COMPARISON:"
echo "======================="

for processes in 2 3 6; do
    logfile="test_${processes}_processes.log"
    if [ -f "$logfile" ]; then
        time=$(grep "Total execution time:" "$logfile" | grep -o '[0-9]\+\.[0-9]\+')
        provinces=$(grep "provinces analyzed" "$logfile" | grep -o '[0-9]\+' | head -1)
        if [ ! -z "$time" ] && [ ! -z "$provinces" ]; then
            avg_time=$(echo "scale=2; $time / $provinces" | bc -l 2>/dev/null || echo "N/A")
            echo "$processes processes: ${time}s total, ${provinces} provinces, ${avg_time}s avg per province"
        else
            echo "$processes processes: Check $logfile for details"
        fi
    fi
done

echo ""
echo "==================================================================="
echo "SPEEDUP ANALYSIS:"
echo "================="

# Calculate speedup (if baseline with 2 processes exists)
baseline_time=""
if [ -f "test_2_processes.log" ]; then
    baseline_time=$(grep "Total execution time:" "test_2_processes.log" | grep -o '[0-9]\+\.[0-9]\+')
fi

if [ ! -z "$baseline_time" ]; then
    echo "Baseline (2 processes): ${baseline_time}s"
    
    for processes in 3 6; do
        logfile="test_${processes}_processes.log"
        if [ -f "$logfile" ]; then
            time=$(grep "Total execution time:" "$logfile" | grep -o '[0-9]\+\.[0-9]\+')
            if [ ! -z "$time" ]; then
                speedup=$(echo "scale=2; $baseline_time / $time" | bc -l 2>/dev/null || echo "N/A")
                efficiency=$(echo "scale=2; $speedup / $processes * 100" | bc -l 2>/dev/null || echo "N/A")
                echo "$processes processes: ${time}s, Speedup: ${speedup}x, Efficiency: ${efficiency}%"
            fi
        fi
    done
else
    echo "Could not calculate speedup - baseline timing not available"
fi

echo ""
echo "==================================================================="
echo "DETAILED LOGS:"
echo "=============="
echo "Check the following files for detailed output:"
echo "- test_2_processes.log"
echo "- test_3_processes.log" 
echo "- test_6_processes.log"
echo "==================================================================="

# Show sample results
echo ""
echo "SAMPLE RESULTS (from 2-process run):"
echo "===================================="
if [ -f "test_2_processes.log" ]; then
    grep -A 20 "Province:" "test_2_processes.log" | head -25
fi

echo ""
echo "Testing complete! Review the log files for full results."