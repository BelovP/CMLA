for file in test_*; do
    printf "Running ${file}... "
    ./$file
    echo "OK"
done