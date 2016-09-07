<?php

try {

    $id  = $_REQUEST['id'];
    $num = $_REQUEST['num'];
    $path = "";
    if (!isset($id))
        throw new Exception("request must specify an id");
    if (strpos($id, '/') !== false)
        $path = $id;
    else {
        $path = "/data/tmp/ncreview/$id/";
        if (!file_exists($path) || !is_readable($path)) {
            $path = "/data/dmf/tmp/ncreview/$id/";
            if (!file_exists($path) || !is_readable($path)) {
                throw new Exception("Cannot find directory $id in any /data/tmp");
            }  
        }
    }
    if (!$num)
        $path .= "ncreview.json";
    else
        $path .= "ncreview.$num.csv";

    if (!file_exists($path) || !is_readable($path)) {
        throw new Exception("File $path is unreadable or does not exist.");
    }
    print file_get_contents($path);

} catch (Exception $e) {
    header($_SERVER['SERVER_PROTOCOL'] . ' 500 Internal Server Error',
true, 500);
    print $e->getMessage();
}

?>