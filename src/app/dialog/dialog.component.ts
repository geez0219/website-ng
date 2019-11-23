import { Component, Inject, Injectable } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { Location, LocationStrategy, PathLocationStrategy } from '@angular/common';
import { WINDOW } from '../window-provider/window-provider.component'

@Component({
  selector: 'app-dialog',
  providers: [Location, {provide: LocationStrategy, useClass: PathLocationStrategy}],
  templateUrl: './dialog.component.html',
  styleUrls: ['./dialog.component.css']
})

export class DialogComponent {
  searchData:any;
  location_data:any;
  constructor(
    public dialogRef: MatDialogRef<DialogComponent>,
    @Inject(WINDOW) private window:Window,
    @Inject(MAT_DIALOG_DATA) public data: any) {
      this.searchData = data;
      console.log("hey");
      console.log("hey");
      console.log(window.location);
      this.location_data = window.location;
    }
}

// export class DialogComponent {
//   searchData:any;
//   location:Location;
//   constructor(
//     public dialogRef: MatDialogRef<DialogComponent>, location:Location,
//     @Inject(MAT_DIALOG_DATA) public data: any) {
//       this.searchData = data;
//       console.log("hey");
//       console.log("hey");
//       console.log(location);
//     }
// }