for file in test_*; do
    printf "Running ${file}... "
    if [ "$1" == "-v" ]; then
        echo
    fi

    ./$file $1
    if [ "$?" == "0" ]; then
        echo "OK"
    else
        break
    fi

    if [ "$1" == "-v" ]; then
        echo
    fi
done