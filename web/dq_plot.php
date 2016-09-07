<?php
$plot = $_REQUEST['plot'];

// check that $plot is a path to a .png
if (preg_match('/^[a-zA-Z0-9\-\.\/]+\.png\z/', $plot) === false) {
	throw new Exception('Directory argument does not match expected pattern');
}

$fp = fopen($plot, 'rb');

header("Content-Type: image/png");
header("Content-Length: " . filesize($plot));

fpassthru($fp);

fclose($fp);
?>