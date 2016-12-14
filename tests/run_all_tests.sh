for file in test_*; do
    printf "Running ${file}... "
    if [ "$1" == "-v" ]; then
        echo
    fi

    ./$file $1
    echo "OK"

    if [ "$1" == "-v" ]; then
        echo
    fi
done