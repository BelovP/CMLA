for file in test_*; do
    echo "Running ${file}..."
    ./$file
done