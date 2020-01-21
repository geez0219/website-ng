import { Component, OnInit, Inject } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material';

export interface DialogData {
}

@Component({
  selector: 'app-search-result',
  templateUrl: './search-result.component.html',
  styleUrls: ['./search-result.component.css']
})
export class SearchResultComponent implements OnInit {
  data: any;
  constructor(
    //public dialogRef: MatDialogRef<SearchResultComponent>,
    //@Inject(MAT_DIALOG_DATA) public data: DialogData
    )
    {}

  onNoClick(): void {

  }

  ngOnInit() {
    console.log(history.state)
    this.data = history.state
  }

}
